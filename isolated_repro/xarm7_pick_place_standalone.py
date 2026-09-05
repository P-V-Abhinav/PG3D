"""Standalone xArm7-gripper pick-and-place repro (v3).

v3 throws away every grasp/gripper/arm choice I had made myself and instead
mirrors, as exactly as a single self-contained file can, the *known-working*
implementation from the senior's cube pick-and-place data generator:

    agents.py::XArm7Gripper          -> agent config (gains, keyframe, mimic,
                                        collision groups, controllers)
    write_xarm7_cube_dataset.py      -> the whole pick cycle (waypoints,
                                        gripper targets, phase structure)
    pg3d/envs/xarm_adapter/motionplanner.py
                                     -> mplib planner setup (SRDF + convex
                                        hull generation, MOVE_GROUP=link_tcp)

Everything below that touches the grasp is copied from those files rather
than invented here. The differences that remain, and why:

  * The agent's `uid` is "isolated_xarm7_gripper" instead of "xarm7_gripper",
    purely so this file can be run in a process that has also imported the
    repo's own agents.py without a duplicate-registration clash. Every
    physical parameter is byte-for-byte the reference's.
  * The env is defined here (table + cube + goal marker) because the
    senior's PG3DReachXArm7CubeV2Env / cube_pick_place_env.py is not in this
    checkout. It is a plain TableSceneBuilder scene, which is what that env
    is too.
  * Legs are planned-and-executed one at a time straight off the live robot
    qpos, instead of the reference's plan-everything-first-then-replay (it
    does that so it can sample N trajectory *families* per cube reset for a
    dataset; there is nothing to sample here). The planner calls, the
    waypoints, and the per-step action format are the same.

WHAT SPECIFICALLY CHANGED FROM MY (BROKEN) v2, ALL SOURCED FROM THE REFERENCE
============================================================================
  * rest keyframe qpos: [0, -0.5, 0, 1.0, 0, 1.2, 0] (v2 used [0,-0.4,0,0.5,0,0.9,0]).
    The whole cycle inherits its TCP orientation from this pose -- see below.
  * gripper_damping 500 (was 2000), gripper_force_limit 50 (was 1),
    gripper_friction 1.0, mimic upper 0.85 (v2 backed it off to 0.84).
  * NO urdf_config friction material on the fingers. My v1/v2 added
    static_friction=2.0 pads; the working reference does not have them, so
    they are gone.
  * GRIPPER CLOSED TARGET IS 0.55, NOT 0.85. This is probably the single
    biggest fix: 0.55 rad is where this gripper actually grips a ~4cm cube.
    Commanding 0.85 (fully closed) means the PD spring is forever driving
    *through* the cube, which is what was launching it. No ramp and no
    stall-detection is needed once the target is right -- the reference just
    holds the grasp qpos for CLOSE_HOLD_STEPS=15 steps at 0.55.
  * arm_stiffness 1000 (was 2000).
  * Control is joint-space `pd_joint_pos` with an 8-dim action
    [7 arm qpos, gripper target], driven by mplib screw plans -- not my
    hand-rolled Cartesian servo. TCP orientation is never commanded
    explicitly: every waypoint pose reuses the quaternion the TCP already
    has at the rest keyframe (`quat = start_tcp_pose[3:7]`), exactly as
    plan_all_family_pick_place does.
  * Waypoints are the reference's: standoff = cube + 3cm, grasp = cube
    center, lift = grasp + 12cm, goal standoff at lift height, place = goal,
    retreat = goal + 12cm, then home.

Run on the server:
    python isolated_repro/xarm7_pick_place_standalone.py --video-out xarm7_repro/run.mp4
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
import trimesh
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *  # noqa: F401,F403 -- controller configs + deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import (
    BaseMotionPlanningSolver,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# ===========================================================================
# Constants -- all copied from write_xarm7_cube_dataset.py
# ===========================================================================

ROBOT_BASE_POSE = sapien.Pose(p=[-0.615, 0.0, 0.0])   # same base pose the repo's xArm7 envs use

GRIPPER_OPEN_Q = 0.0
GRIPPER_CLOSED_Q = 0.55       # reference value -- NOT 0.85. See module docstring.

APPROACH_STANDOFF = 0.03      # 3cm standoff above the cube before descending to grasp
LIFT_HEIGHT = 0.12            # lift straight up after closing
CLOSE_HOLD_STEPS = 15         # steps holding the grasp pose while the gripper closes
OPEN_HOLD_STEPS = 10          # steps holding the placed pose while the gripper opens

CUBE_HALF_SIZE = (0.02, 0.02, 0.03)   # reference DEFAULT_CUBE_HALF_SIZE
CUBE_SPAWN_XY = (-0.25, 0.0)
PLACE_XY = (-0.25, 0.2)

SUCCESS_LIFT_FRACTION = 0.5           # cube must rise >= this fraction of LIFT_HEIGHT
PLACE_XY_SUCCESS_TOLERANCE = 0.03


# ===========================================================================
# URDF resolution
#
# mplib ignores absolute mesh paths and prepends the URDF's own directory,
# so the URDF must keep RELATIVE "meshes/..." references with a sibling
# `meshes` symlink -- exactly the arrangement agents.py's
# _build_xarm7_gripper_colored_urdf sets up. (v2 rewrote the paths to
# absolute, which works for SAPIEN but breaks mplib.) The committed colored
# URDF is copied here verbatim and a fresh symlink is pointed at this
# server's real meshes directory.
# ===========================================================================

_XARM7_GRIPPER_MESHES_DIR = Path(
    "/home/cross-emb/abhinav.pv/success/.venv/lib/python3.10/site-packages/mani_skill/assets/robots/xarm7/meshes"
)


def _resolve_gripper_urdf() -> str:
    here = Path(__file__).resolve().parent
    src = (
        here.parent / "pg3d" / "envs" / "xarm_adapter" / "assets" / "xarm7_with_gripper_colored.urdf"
    )
    if not src.exists():
        raise FileNotFoundError(f"expected the committed URDF at {src}")
    if not _XARM7_GRIPPER_MESHES_DIR.is_dir():
        raise FileNotFoundError(f"_XARM7_GRIPPER_MESHES_DIR does not exist: {_XARM7_GRIPPER_MESHES_DIR}")

    meshes_link = here / "meshes"
    target = str(_XARM7_GRIPPER_MESHES_DIR)
    if meshes_link.is_symlink() and os.readlink(str(meshes_link)) != target:
        meshes_link.unlink()
    if not meshes_link.exists() and not meshes_link.is_symlink():
        os.symlink(target, str(meshes_link))

    dst = here / "xarm7_with_gripper_isolated.urdf"
    dst.write_text(src.read_text())   # verbatim: relative "meshes/..." refs preserved
    return str(dst)


_XARM7_GRIPPER_URDF = _resolve_gripper_urdf()


# ===========================================================================
# 1. Agent -- config copied verbatim from agents.py::XArm7Gripper
# ===========================================================================

@register_agent()
class IsolatedXArm7Gripper(BaseAgent):
    """xArm7 + xArm parallel-jaw gripper. Physical config identical to
    agents.py::XArm7Gripper (only `uid` differs, to avoid a registration
    clash if the repo's own agents.py is imported in the same process)."""

    uid = "isolated_xarm7_gripper"
    urdf_path = _XARM7_GRIPPER_URDF

    gripper_joint_names = [
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    ]

    gripper_stiffness = 1e5
    gripper_damping = 500
    gripper_force_limit = 50
    gripper_friction = 1.0

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([
                0.0, -0.5, 0.0, 1.0, 0.0, 1.2, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # gripper open (0.0 = open, 0.85 = closed)
            ]),
        ),
    )

    arm_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ]

    arm_stiffness = 1000
    arm_damping = [100, 100, 100, 100, 100, 100, 100]
    arm_friction = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    arm_force_limit = 100

    ee_link_name = "link_tcp"

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :7]  # arm joints only
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def _after_init(self):
        self.finger1_link = self.robot.links_map["left_finger"]
        self.finger2_link = self.robot.links_map["right_finger"]
        self.tcp = self.robot.links_map[self.ee_link_name]

    def _after_loading_articulation(self):
        # This gripper's mimic joints have NO shared link tying the two
        # branches together (unlike xarm6_robotiq's four-bar loop closure
        # via scene.create_drive). left_finger and left_inner_knuckle sit
        # geometrically close in the real assembly, so disable collisions
        # among all gripper-internal links to stop self-contact noise from
        # fighting the PD mimic targets and causing asymmetric closing.
        gripper_links = [
            "xarm_gripper_base_link",
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
            "link7",  # adjacent arm link
        ]
        for link_name in gripper_links:
            link = self.robot.links_map.get(link_name)
            if link is not None:
                link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(  # noqa: F405
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        arm_pd_joint_delta_pos = PDJointPosControllerConfig(  # noqa: F405
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        mimic_config = {
            "left_finger_joint": {"joint": "drive_joint", "multiplier": 1.0, "offset": 0.0},
            "left_inner_knuckle_joint": {"joint": "drive_joint", "multiplier": 1.0, "offset": 0.0},
            "right_outer_knuckle_joint": {"joint": "drive_joint", "multiplier": 1.0, "offset": 0.0},
            "right_finger_joint": {"joint": "drive_joint", "multiplier": 1.0, "offset": 0.0},
            "right_inner_knuckle_joint": {"joint": "drive_joint", "multiplier": 1.0, "offset": 0.0},
        }

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(  # noqa: F405
            self.gripper_joint_names,
            lower=0.0,
            upper=0.85,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
            mimic=mimic_config,
        )

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
        )

        return deepcopy_dict(controller_configs)  # noqa: F405


# ===========================================================================
# 2. Env -- table + cube + goal marker.
# ===========================================================================

@register_env("Isolated-XArm7-PickCube-v0", max_episode_steps=100_000)
class IsolatedXArm7PickCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["isolated_xarm7_gripper"]

    def __init__(self, *args: Any, robot_uids: str = "isolated_xarm7_gripper", **kwargs: Any) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, ROBOT_BASE_POSE)

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=list(CUBE_HALF_SIZE))
        builder.add_box_visual(
            half_size=list(CUBE_HALF_SIZE),
            material=sapien.render.RenderMaterial(base_color=[0.1, 0.8, 0.1, 1.0]),
        )
        builder.initial_pose = sapien.Pose(p=[CUBE_SPAWN_XY[0], CUBE_SPAWN_XY[1], CUBE_HALF_SIZE[2]])
        self.cube = builder.build(name="cube")

        self.goal_marker = actors.build_box(
            self.scene,
            half_sizes=[0.035, 0.035, 0.001],
            color=[1.0, 0.2, 0.2, 1.0],
            name="goal_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[PLACE_XY[0], PLACE_XY[1], 0.001]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)

            # xArm7 at the URDF-default zeros qpos sits near a singularity and
            # the PD controller can blow up -- always reset to "rest".
            rest_qpos = self.agent.keyframes["rest"].qpos
            qpos = torch.tensor(rest_qpos, dtype=torch.float32).unsqueeze(0).expand(b, -1).clone()
            self.agent.reset(qpos)

            cube_xyz = torch.tensor(
                [CUBE_SPAWN_XY[0], CUBE_SPAWN_XY[1], CUBE_HALF_SIZE[2]], dtype=torch.float32
            ).unsqueeze(0).expand(b, -1).clone()
            self.cube.set_pose(Pose.create_from_pq(cube_xyz))

            goal_xyz = torch.tensor(
                [PLACE_XY[0], PLACE_XY[1], 0.001], dtype=torch.float32
            ).unsqueeze(0).expand(b, -1).clone()
            self.goal_marker.set_pose(Pose.create_from_pq(goal_xyz))

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return []

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(eye=[0.4, -0.55, 0.55], target=[-0.25, 0.05, 0.05])
        return CameraConfig("render_camera", pose, 640, 480, 1.0, 0.01, 100)

    def evaluate(self) -> dict[str, torch.Tensor]:
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}


# ===========================================================================
# 3. mplib planner -- setup copied from
#    pg3d/envs/xarm_adapter/motionplanner.py (XArm7GripperMotionPlanningSolver)
# ===========================================================================

def _watertight_convex_hull(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Convex hull guaranteed watertight. trimesh's own convex_hull merges
    coplanar faces and can leave open edges, which mplib hard-errors on."""
    from scipy.spatial import ConvexHull

    pts = np.asarray(mesh.vertices, dtype=np.float64)
    hull = ConvexHull(pts)
    used = np.unique(hull.simplices)
    remap = {int(old): i for i, old in enumerate(used)}
    faces = np.array([[remap[int(v)] for v in simplex] for simplex in hull.simplices])
    out = trimesh.Trimesh(vertices=pts[used], faces=faces, process=False)
    trimesh.repair.fix_normals(out)
    return out


def _ensure_convex_collision_meshes(urdf_path: str) -> None:
    """Generate/repair the `<mesh>.convex.stl` files mplib expects."""
    urdf_dir = os.path.dirname(urdf_path)
    root = ET.parse(urdf_path).getroot()
    seen: set[str] = set()
    for collision in root.iter("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        rel = mesh.get("filename")
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        src = os.path.normpath(os.path.join(urdf_dir, rel))
        dst = f"{src}.convex.stl"
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and trimesh.load(dst, force="mesh").is_watertight:
            continue
        hull = _watertight_convex_hull(trimesh.load(src, force="mesh"))
        hull.export(dst)


def _gripper_rigid_cluster(root: ET.Element) -> list[str]:
    """Links sharing a rigid body with the gripper (gripper subtree + the
    fixed-joint ancestors it is bolted to). All pairs among these get
    disabled in the SRDF, else mplib sees a permanently self-colliding robot
    and every IK/plan fails."""
    joints = root.findall("joint")
    fix = next((j for j in joints if j.get("name") == "gripper_fix"), None)
    if fix is None or fix.find("child") is None:
        return []
    base = fix.find("child").get("link")

    children: dict[str, list[str]] = {}
    parent_joint: dict[str, ET.Element] = {}
    for j in joints:
        p, c = j.find("parent"), j.find("child")
        if p is not None and c is not None:
            children.setdefault(p.get("link"), []).append(c.get("link"))
            parent_joint[c.get("link")] = j

    cluster: list[str] = []
    stack = [base]
    while stack:
        link = stack.pop()
        cluster.append(link)
        stack.extend(children.get(link, []))

    link = base
    while link in parent_joint and parent_joint[link].get("type") == "fixed":
        link = parent_joint[link].find("parent").get("link")
        if link not in cluster:
            cluster.append(link)

    return cluster


def _ensure_srdf(urdf_path: str) -> str:
    """Generate/repair an SRDF disabling the self-collision pairs mplib must
    ignore (adjacent pairs + the gripper rigid cluster)."""
    srdf_path = urdf_path.replace(".urdf", ".srdf")
    root = ET.parse(urdf_path).getroot()
    robot_name = root.get("name", "robot")

    required: dict[frozenset[str], str] = {}

    def need(a: str, b: str, reason: str) -> None:
        if a and b and a != b:
            required.setdefault(frozenset((a, b)), reason)

    for j in root.findall("joint"):
        p, c = j.find("parent"), j.find("child")
        if p is not None and c is not None:
            need(p.get("link"), c.get("link"), "Adjacent")

    grip = _gripper_rigid_cluster(root)
    for i in range(len(grip)):
        for k in range(i + 1, len(grip)):
            need(grip[i], grip[k], "Gripper")

    existing: dict[frozenset[str], str] = {}
    if os.path.exists(srdf_path):
        for dc in ET.parse(srdf_path).getroot().findall("disable_collisions"):
            l1, l2 = dc.get("link1"), dc.get("link2")
            if l1 and l2:
                existing[frozenset((l1, l2))] = dc.get("reason", "Never")
        if all(key in existing for key in required):
            return srdf_path

    merged = {**required, **existing}
    lines = [f'<robot name="{robot_name}">']
    for key, reason in merged.items():
        a, b = tuple(key) if len(key) == 2 else (next(iter(key)), next(iter(key)))
        lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{reason}"/>')
    lines.append("</robot>\n")
    with open(srdf_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srdf_path


class IsolatedXArm7GripperMotionPlanningSolver(BaseMotionPlanningSolver):
    """mplib planner for this agent (TCP = link_tcp, 7-DOF chain).

    setup_planner is copied from XArm7MotionPlanningSolverBase. follow_path
    is overridden to append the gripper column to mplib's 7-dim waypoints
    (the base class's version emits arm-only actions) and to capture render
    frames -- the same 8-dim [arm qpos, gripper target] action format
    write_xarm7_cube_dataset.py's _format_arm_gripper_action produces.
    """

    MOVE_GROUP = "link_tcp"

    def __init__(self, *args, visualize_target_grasp_pose: bool = False, frames: list | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gripper_state = GRIPPER_OPEN_Q
        self.frames = frames

    def setup_planner(self):
        import mplib

        urdf = self.env_agent.urdf_path
        _ensure_convex_collision_meshes(urdf)
        srdf = _ensure_srdf(urdf)
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=urdf,
            srdf=srdf,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner

    def _capture(self):
        if self.frames is not None:
            self.frames.append(_to_numpy_frame(self.base_env.render()))

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        obs = reward = terminated = truncated = info = None
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            action = np.hstack([qpos[:7], self.gripper_state]).astype(np.float32)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            self._capture()
        return obs, reward, terminated, truncated, info

    def hold(self, steps: int, gripper_target: float):
        """Hold the current arm qpos while the gripper drives to
        `gripper_target` -- the reference's "close"/"open" phases, which are
        just np.repeat(grasp_qpos, hold_steps) at the new gripper value."""
        self.gripper_state = gripper_target
        qpos = self.robot.get_qpos().cpu().numpy()[0][:7]
        for _ in range(steps):
            action = np.hstack([qpos, gripper_target]).astype(np.float32)
            self.env.step(action)
            self.elapsed_steps += 1
            self._capture()


# ===========================================================================
# 4. Pick cycle -- waypoints/phases copied from
#    write_xarm7_cube_dataset.py::plan_all_family_pick_place
# ===========================================================================

def _to_numpy_frame(raw: Any) -> np.ndarray:
    arr = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def _log(env: Any, label: str) -> None:
    u = env.unwrapped
    tcp = u.agent.tcp_pose.p[0].cpu().numpy()
    qvel = u.agent.robot.get_qvel()[0, :7].cpu().numpy()
    drive = u.agent.robot.active_joints_map["drive_joint"]
    print(
        f"  [{label}] tcp={np.round(tcp, 4).tolist()}  "
        f"cube={np.round(u.cube.pose.p[0].cpu().numpy(), 4).tolist()}  "
        f"drive_q={float(drive.qpos[0]):.4f}  max|qvel|={float(np.max(np.abs(qvel))):.3f}",
        flush=True,
    )


def _pose(position: np.ndarray, quat: np.ndarray) -> sapien.Pose:
    return sapien.Pose(p=np.asarray(position, dtype=np.float64), q=np.asarray(quat, dtype=np.float64))


def _move(planner: Any, position: np.ndarray, quat: np.ndarray, label: str) -> bool:
    res = planner.move_to_pose_with_screw(_pose(position, quat))
    ok = res != -1
    print(f"  [{label}] screw plan {'OK' if ok else 'FAILED'} -> {np.round(position, 4).tolist()}", flush=True)
    return ok


def _save_video(frames: list[np.ndarray], path: str, fps: int = 60) -> None:
    if not frames:
        print("[warn] no frames recorded.")
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] imageio not installed; skipping video save.")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"saved video: {path}  ({len(frames)} frames)")


def main(argv: list[str] | None = None) -> int:
    import gymnasium as gym

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gripper-closed-q", type=float, default=GRIPPER_CLOSED_Q)
    parser.add_argument("--close-hold-steps", type=int, default=CLOSE_HOLD_STEPS)
    parser.add_argument("--open-hold-steps", type=int, default=OPEN_HOLD_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-out", type=str, default="./isolated_repro_output.mp4")
    args = parser.parse_args(argv)

    env = gym.make(
        "Isolated-XArm7-PickCube-v0",
        obs_mode="none",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        enable_shadow=True,
    )
    env.reset(seed=args.seed)
    u = env.unwrapped

    frames: list[np.ndarray] = [_to_numpy_frame(env.render())]
    planner = IsolatedXArm7GripperMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        frames=frames,
    )

    # TCP orientation for every waypoint = the orientation the TCP already
    # has at the rest keyframe, held constant for the whole cycle. This is
    # `quat = start_tcp_pose[3:7]` in plan_all_family_pick_place.
    start_tcp = u.agent.tcp_pose
    quat = start_tcp.q[0].cpu().numpy().astype(np.float64)
    home_xyz = start_tcp.p[0].cpu().numpy().astype(np.float64)
    cube_pos = u.cube.pose.p[0].cpu().numpy().astype(np.float64)
    goal_xyz = np.array([PLACE_XY[0], PLACE_XY[1], CUBE_HALF_SIZE[2]], dtype=np.float64)
    spawn_z = float(cube_pos[2])

    pick_standoff_xyz = cube_pos + np.array([0, 0, APPROACH_STANDOFF])
    pick_xyz = cube_pos.copy()
    lift_xyz = pick_xyz + np.array([0, 0, LIFT_HEIGHT])
    goal_standoff_xyz = np.array([goal_xyz[0], goal_xyz[1], lift_xyz[2]])
    retreat_xyz = goal_xyz + np.array([0, 0, LIFT_HEIGHT])

    print(f"home tcp={np.round(home_xyz,4).tolist()}  quat={np.round(quat,4).tolist()}")
    print(f"cube={np.round(cube_pos,4).tolist()}  goal={np.round(goal_xyz,4).tolist()}")
    _log(env, "reset")

    planner.gripper_state = GRIPPER_OPEN_Q

    print("\n--- approach ---")
    if not _move(planner, pick_standoff_xyz, quat, "approach"):
        return 1
    _log(env, "post-approach")

    print("\n--- descend ---")
    if not _move(planner, pick_xyz, quat, "descend"):
        return 1
    _log(env, "post-descend")

    print(f"\n--- close (hold {args.close_hold_steps} steps at q={args.gripper_closed_q}) ---")
    planner.hold(args.close_hold_steps, args.gripper_closed_q)
    _log(env, "post-close")

    print("\n--- lift ---")
    if not _move(planner, lift_xyz, quat, "lift"):
        return 1
    _log(env, "post-lift")
    lifted_cube_z = float(u.cube.pose.p[0, 2].item())

    print("\n--- transport ---")
    if not _move(planner, goal_standoff_xyz, quat, "transport"):
        return 1
    _log(env, "post-transport")
    print(
        f"  cube z after lift {lifted_cube_z:.4f} -> after transport "
        f"{float(u.cube.pose.p[0,2].item()):.4f} (big drop = slipped mid-transit)"
    )

    print("\n--- place_descend ---")
    if not _move(planner, goal_xyz, quat, "place_descend"):
        return 1
    _log(env, "post-place-descend")

    print(f"\n--- open (hold {args.open_hold_steps} steps at q={GRIPPER_OPEN_Q}) ---")
    planner.hold(args.open_hold_steps, GRIPPER_OPEN_Q)
    _log(env, "post-open")

    print("\n--- retreat ---")
    _move(planner, retreat_xyz, quat, "retreat")
    _log(env, "post-retreat")

    print("\n--- home ---")
    _move(planner, home_xyz, quat, "home")
    _log(env, "final")

    final_cube = u.cube.pose.p[0].cpu().numpy()
    place_distance = float(np.linalg.norm(final_cube[:2] - goal_xyz[:2]))
    max_lift = float(lifted_cube_z - spawn_z)
    print(f"\nfinal cube={np.round(final_cube,4).tolist()}  goal={np.round(goal_xyz,4).tolist()}")
    print(f"place_distance={place_distance:.4f} m (success < {PLACE_XY_SUCCESS_TOLERANCE})")
    print(f"lift delta={max_lift:.4f} m (success >= {SUCCESS_LIFT_FRACTION * LIFT_HEIGHT:.4f})")
    print(
        "RESULT: "
        + ("SUCCESS" if (place_distance < PLACE_XY_SUCCESS_TOLERANCE
                         and max_lift >= SUCCESS_LIFT_FRACTION * LIFT_HEIGHT) else "FAILED")
    )

    _save_video(frames, args.video_out)
    planner.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
