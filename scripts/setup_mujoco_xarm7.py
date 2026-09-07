"""Fetch the MuJoCo Menagerie xArm7+gripper MJCF and patch it for ManiSkill.

Run once per machine before using the ``xarm7_gripper`` agent::

    python -m scripts.setup_mujoco_xarm7

What this does
--------------
1. Downloads ``ufactory_xarm7/`` from google-deepmind/mujoco_menagerie into
   ``pg3d/envs/xarm_adapter/assets/mujoco_xarm7/`` (the XML plus every mesh the
   XML actually references -- parsed out of ``<asset>``, not hardcoded).
2. Writes ``xarm7_maniskill.xml`` beside the original, patched in two ways that
   SAPIEN's MJCF loader requires. Both are documented at the patch sites below.

The patched file is written NEXT TO the original on purpose: MuJoCo resolves
``<compiler meshdir="assets">`` relative to the XML's own directory, so the
patched copy must stay a sibling of ``assets/`` to find its meshes.

Why a patch is needed at all -- SAPIEN/ManiSkill MJCF loader gaps
-----------------------------------------------------------------
ManiSkill's MJCF support is real but partial (see
https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html
"ManiSkill supports importing Mujoco's MJCF format ... although not all features
are supported"). The gaps that bite this particular model:

* ``<site>`` elements do NOT become links. The Menagerie model defines the tool
  centre point as ``<site name="link_tcp" pos="0 0 .172"/>``, but the pg3d reach
  env and the mplib planner both need ``link_tcp`` to be a real *link*
  (``sapien_utils.get_obj_by_name(robot.get_links(), "link_tcp")``, and mplib's
  ``MOVE_GROUP``). Patch #1 promotes the site to a massless welded body.
* ``<equality>`` constraints are dropped. PhysX articulations are strictly
  trees, so the two ``<connect>`` constraints that close the gripper's
  parallelogram four-bar linkages cannot survive the import. They are restored
  at runtime with ``scene.create_drive`` in ``XArm7Gripper`` -- the same
  mechanism that already stood in for URDF's inability to express the loop.
* ``<actuator>``/``<tendon>`` are dropped ("Importing motors and solver
  configurations" is listed as unsupported). ManiSkill's own controller configs
  replace them, so this is expected rather than a problem.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO = "google-deepmind/mujoco_menagerie"
_MODEL_DIR = "ufactory_xarm7"
_RAW = f"https://raw.githubusercontent.com/{_REPO}/main"

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "pg3d" / "envs" / "xarm_adapter" / "assets"
DEST = ASSETS_ROOT / "mujoco_xarm7"

SOURCE_XML = "xarm7.xml"
PATCHED_XML = "xarm7_maniskill.xml"

# The site the Menagerie model uses for the TCP, and the body we promote it to.
_TCP_NAME = "link_tcp"
_TCP_PARENT = "xarm_gripper_base_link"


def _download(rel_path: str, dest: Path) -> None:
    url = f"{_RAW}/{_MODEL_DIR}/{rel_path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            dest.write_bytes(resp.read())
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def _fetch_via_git(dest: Path) -> bool:
    """Sparse-clone just ``ufactory_xarm7/``. Returns False if git is unavailable."""
    if shutil.which("git") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                 f"https://github.com/{_REPO}.git", tmp],
                check=True, capture_output=True,
            )
            subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", _MODEL_DIR],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - network
            print(f"  git sparse-clone failed ({exc.stderr.decode()[:200]}), "
                  "falling back to direct download", file=sys.stderr)
            return False
        src = Path(tmp) / _MODEL_DIR
        if not (src / SOURCE_XML).exists():
            return False
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    return True


def _fetch_via_http(dest: Path) -> None:
    """Download the XML, then exactly the meshes its <asset> block references."""
    dest.mkdir(parents=True, exist_ok=True)
    _download(SOURCE_XML, dest / SOURCE_XML)
    root = ET.parse(dest / SOURCE_XML).getroot()

    compiler = root.find("compiler")
    meshdir = (compiler.get("meshdir") if compiler is not None else None) or "."

    files = [m.get("file") for m in root.iter("mesh") if m.get("file")]
    files += [t.get("file") for t in root.iter("texture") if t.get("file")]
    print(f"  {len(files)} asset file(s) referenced by {SOURCE_XML}")
    for f in files:
        rel = f"{meshdir}/{f}" if meshdir != "." else f
        _download(rel, dest / rel)


def _patch(dest: Path) -> Path:
    """Write the ManiSkill-compatible XML and return its path."""
    src = dest / SOURCE_XML
    tree = ET.parse(src)
    root = tree.getroot()

    # ---- Patch #1: promote the link_tcp <site> to a real body ------------------
    # SAPIEN's MJCF loader does not turn sites into links, but pg3d needs link_tcp
    # as a link (agent.tcp / ee_link_name / mplib MOVE_GROUP). A MuJoCo body with
    # no joint is welded to its parent, which is precisely the "fixed joint" the
    # equivalent URDF uses -- so this is a faithful translation, not a fudge.
    #
    # The tiny explicit <inertial> is deliberate: MuJoCo tolerates a massless
    # welded body, but a zero-mass link is a hazard for downstream inertia math,
    # and SAPIEN's loader is happier with an explicit one. 1e-6 kg is ~6 orders of
    # magnitude below the lightest real gripper link, so it perturbs nothing.
    #
    # VERIFY THIS FIRST if env construction fails with link_tcp not found: some
    # loaders fuse a jointless body into its parent rather than emitting a
    # separate link. If that happens here, give the body a locked hinge --
    #     <joint name="joint_tcp" type="hinge" axis="0 0 1" range="0 0"/>
    # -- which forces a distinct link. That adds a 14th (immobile) DOF, so
    # XArm7Gripper.keyframes and any qpos slicing must grow by one; prefer the
    # jointless form and only fall back if the loader actually fuses it.
    parent = next((b for b in root.iter("body") if b.get("name") == _TCP_PARENT), None)
    if parent is None:
        raise RuntimeError(f"{src}: no body named {_TCP_PARENT!r} -- upstream model changed")

    site = next((s for s in parent.findall("site") if s.get("name") == _TCP_NAME), None)
    tcp_pos = site.get("pos") if site is not None else "0 0 0.172"
    if site is None:
        print(f"  warning: no <site name={_TCP_NAME!r}>; defaulting TCP to {tcp_pos}",
              file=sys.stderr)

    if not any(b.get("name") == _TCP_NAME for b in parent.findall("body")):
        body = ET.SubElement(parent, "body", {"name": _TCP_NAME, "pos": tcp_pos})
        ET.SubElement(body, "inertial", {
            "pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-9 1e-9 1e-9",
        })

    # ---- Patch #2: drop <actuator>/<tendon>/<equality> explicitly --------------
    # ManiSkill ignores these anyway, but leaving them in makes the import look
    # lossless when it is not. Removing them makes the loss explicit and keeps the
    # loader from warning on every construction. The physics they encoded is
    # re-created in XArm7Gripper:
    #   <equality connect>  -> scene.create_drive() loop closure
    #   <equality joint> +
    #   <tendon>/<actuator> -> PDJointPosMimicControllerConfig
    for tag in ("actuator", "tendon", "equality"):
        for elem in root.findall(tag):
            root.remove(elem)
            print(f"  removed <{tag}> (unsupported by the SAPIEN MJCF loader; "
                  "re-created in the agent)")

    out = dest / PATCHED_XML
    tree.write(out, xml_declaration=True, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the model is already present")
    args = ap.parse_args()

    if (DEST / SOURCE_XML).exists() and not args.force:
        print(f"{DEST/SOURCE_XML} already present (use --force to re-download)")
    else:
        print(f"Fetching {_REPO}/{_MODEL_DIR} -> {DEST}")
        if not _fetch_via_git(DEST):
            _fetch_via_http(DEST)

    out = _patch(DEST)
    print(f"\nWrote {out}")

    missing = [
        m.get("file") for m in ET.parse(out).getroot().iter("mesh")
        if m.get("file") and not (DEST / "assets" / m.get("file")).exists()
    ]
    if missing:
        print(f"WARNING: {len(missing)} mesh(es) missing: {missing[:5]}", file=sys.stderr)
        return 1
    print("All referenced meshes present. Agent uid: xarm7_gripper (env ids unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
