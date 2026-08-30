"""Visualize / verify the custom xArm7-with-gripper coordinate frames (incl. TCP).

This is both a **headless pytest** (frame-consistency assertions that skip when the
simulator is unavailable) and a **GUI visualizer** you run directly to eyeball the
frames in the SAPIEN viewer.

Why this script:
    We want to see exactly where `link_tcp` is on the custom xArm7-gripper, because
    `link_tcp` is the frame the IK planner targets and the goal position is measured
    against -- if it were misplaced, IK and goal-reaching would be silently wrong.
    The script attaches SAPIEN's *native* coordinate frame (an RGB arrow triad, via
    `viewer.add_coordinate_frame`) directly to `link_tcp` (and optionally every arm
    link + base), re-posing it each frame. SAPIEN's built-in "Show Joint Axes" toggle
    only shows one *selected* link at a time and skips fixed links like `link_tcp`,
    which is why we attach the frame explicitly here.

GUI usage (needs a display + the maniskill/viz extras):
    uv run python tests/test_xarm7_gripper_frames.py                 # link_tcp + all links, sweep
    uv run python tests/test_xarm7_gripper_frames.py --tcp-only      # only the link_tcp frame
    uv run python tests/test_xarm7_gripper_frames.py --tcp-only --no-sweep --hold 60
    uv run python tests/test_xarm7_gripper_frames.py --tcp-length 0.12

Headless check (CI-safe; auto-skips if ManiSkill/SAPIEN missing):
    uv run pytest tests/test_xarm7_gripper_frames.py -q
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pytest

ENV_ID = "PG3DReach-XArm7-Gripper-Workspace-v0"
ROBOT_UID = "xarm7_gripper"
TCP_LINK_NAME = "link_tcp"
ARM_LINK_NAMES = [f"link{i}" for i in range(1, 8)]  # link1..link7


# --------------------------------------------------------------------------- #
# Simulator import guard (keeps the module importable / test-collectable on CPU
# boxes without ManiSkill).
# --------------------------------------------------------------------------- #
def _import_sim():
    try:
        import gymnasium as gym  # noqa: F401
        import mani_skill.envs  # noqa: F401
        import sapien  # noqa: F401

        from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
    except Exception as exc:  # pragma: no cover - only on CPU-only envs
        return None, str(exc)
    return (gym, sapien, register_pg3d_xarm7_gripper_reach_envs), None


def _to_np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _link_raw_pose(link) -> np.ndarray:
    """Return a link's world pose as [x,y,z, qw,qx,qy,qz] (SAPIEN layout)."""
    return _to_np(link.pose.raw_pose).reshape(-1, 7)[0].astype(np.float64)


def _get_link(robot, name):
    for link in robot.get_links():
        if link.get_name() == name:
            return link
    have = [link.get_name() for link in robot.get_links()]
    raise KeyError(f"link {name!r} not found; have {have}")


def _quat_angular_distance_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    a = np.asarray(q1, float).reshape(4)
    b = np.asarray(q2, float).reshape(4)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def _set_arm_qpos(robot, qpos_flat: np.ndarray) -> None:
    """Set qpos handling both [D] and [1, D] backends."""
    two_d = _to_np(robot.get_qpos()).ndim == 2
    robot.set_qpos(qpos_flat.reshape(1, -1) if two_d else qpos_flat)


# --------------------------------------------------------------------------- #
# Headless test: the reported TCP (`agent.tcp_pose`) must equal the FK pose of
# the `link_tcp` link across several arm configurations.
# --------------------------------------------------------------------------- #
def test_xarm7_gripper_tcp_matches_link_tcp():
    mods, err = _import_sim()
    if mods is None:
        pytest.skip(f"ManiSkill/SAPIEN unavailable: {err}")
    gym, _sapien, register = mods
    register()

    env = gym.make(
        ENV_ID,
        obs_mode="state",
        control_mode="pd_joint_pos",
        robot_uids=ROBOT_UID,
        num_envs=1,
        render_mode=None,
        sim_backend="auto",
    )
    try:
        env.reset(seed=0)
        agent = env.unwrapped.agent
        robot = agent.robot
        tcp_link = _get_link(robot, TCP_LINK_NAME)

        rng = np.random.default_rng(0)
        qpos0 = _to_np(robot.get_qpos()).reshape(-1)
        for trial in range(5):
            q = qpos0.copy()
            # Perturb the 7 arm joints only; leave the gripper mimic joints as-is.
            q[:7] = q[:7] + rng.uniform(-0.4, 0.4, size=7)
            _set_arm_qpos(robot, q)

            reported = _to_np(agent.tcp_pose.raw_pose).reshape(-1, 7)[0].astype(np.float64)
            fk = _link_raw_pose(tcp_link)

            pos_err = float(np.linalg.norm(reported[:3] - fk[:3]))
            rot_err = _quat_angular_distance_deg(reported[3:7], fk[3:7])
            assert pos_err < 1e-4, f"trial {trial}: TCP pos != link_tcp FK ({pos_err:.2e} m)"
            assert rot_err < 1e-2, f"trial {trial}: TCP rot != link_tcp FK ({rot_err:.3e} deg)"
    finally:
        env.close()


def test_xarm7_gripper_has_expected_links():
    mods, err = _import_sim()
    if mods is None:
        pytest.skip(f"ManiSkill/SAPIEN unavailable: {err}")
    gym, _sapien, register = mods
    register()
    env = gym.make(ENV_ID, obs_mode="state", robot_uids=ROBOT_UID, num_envs=1, render_mode=None)
    try:
        env.reset(seed=0)
        names = {link.get_name() for link in env.unwrapped.agent.robot.get_links()}
        assert TCP_LINK_NAME in names, f"{TCP_LINK_NAME} missing; have {sorted(names)}"
        for arm_link in ARM_LINK_NAMES:
            assert arm_link in names, f"{arm_link} missing; have {sorted(names)}"
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# GUI visualizer (run as a script). Attaches SAPIEN's *native* coordinate frame
# (viewer.add_coordinate_frame -> RGB arrow triad) to `link_tcp` and, optionally,
# each arm link + base, re-posing them every frame so you can see exactly where
# `link_tcp` sits on the robot -- this is the frame the IK planner targets and
# the goal position is measured against.
# --------------------------------------------------------------------------- #
def run_gui(argv=None) -> int:
    parser = argparse.ArgumentParser(description="xArm7-gripper link_tcp frame visualizer")
    parser.add_argument("--tcp-length", type=float, default=0.10, help="TCP frame arrow length (m)")
    parser.add_argument(
        "--tcp-radius", type=float, default=0.006, help="TCP frame arrow thickness (m)"
    )
    parser.add_argument("--link-length", type=float, default=0.05, help="per-link frame length (m)")
    parser.add_argument(
        "--link-radius", type=float, default=0.004, help="per-link frame thickness (m)"
    )
    parser.add_argument("--sweep-steps", type=int, default=240, help="joint-sweep frames")
    parser.add_argument("--hold", type=float, default=0.0, help="seconds to hold at rest first")
    parser.add_argument("--step-delay", type=float, default=0.02, help="sleep per frame (s)")
    parser.add_argument(
        "--tcp-only",
        action="store_true",
        help="attach a frame only to link_tcp (skip per-link and base frames)",
    )
    parser.add_argument(
        "--sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sweep the arm joints so you can see link_tcp track the flange",
    )
    args = parser.parse_args(argv)

    mods, err = _import_sim()
    if mods is None:
        print(f"Cannot open GUI: ManiSkill/SAPIEN unavailable: {err}", file=sys.stderr)
        print(
            "Install: uv sync --extra cu129 --extra maniskill --extra viz --group dev",
            file=sys.stderr,
        )
        return 2
    gym, sapien, register = mods
    register()

    env = gym.make(
        ENV_ID,
        obs_mode="state",
        control_mode="pd_joint_pos",
        robot_uids=ROBOT_UID,
        num_envs=1,
        render_mode="human",
        sim_backend="auto",
        render_backend="gpu",
    )
    try:
        env.reset(seed=0)
        viewer = env.render()  # human render returns the SAPIEN Viewer
        if not hasattr(viewer, "add_coordinate_frame"):
            # Some ManiSkill versions stash the viewer on the env instead.
            viewer = getattr(env.unwrapped, "_viewer", None) or getattr(
                env.unwrapped, "viewer", None
            )
        if viewer is None or not hasattr(viewer, "add_coordinate_frame"):
            print(
                "This SAPIEN viewer has no add_coordinate_frame; cannot attach native frames.",
                file=sys.stderr,
            )
            return 3
        unwrapped = env.unwrapped
        robot = unwrapped.agent.robot

        # Built-in world-origin frame for a fixed reference (best-effort).
        try:
            if hasattr(viewer, "control_window"):
                viewer.control_window.show_origin_frame = True
        except Exception:
            pass

        # Which links get a native coordinate frame. link_tcp is always included
        # and drawn larger (it's the frame we care about).
        frame_specs = []  # (link, node, is_tcp)
        tcp_link = _get_link(robot, TCP_LINK_NAME)
        tcp_node = viewer.add_coordinate_frame(
            sapien.Pose(), length=args.tcp_length, radius=args.tcp_radius
        )
        frame_specs.append((tcp_link, tcp_node, True))

        if not args.tcp_only:
            base_link = robot.get_links()[0]
            base_node = viewer.add_coordinate_frame(
                sapien.Pose(), length=args.link_length, radius=args.link_radius
            )
            frame_specs.append((base_link, base_node, False))
            for name in ARM_LINK_NAMES:
                link = _get_link(robot, name)
                node = viewer.add_coordinate_frame(
                    sapien.Pose(), length=args.link_length, radius=args.link_radius
                )
                frame_specs.append((link, node, False))

        def sync_frames():
            for link, node, _is_tcp in frame_specs:
                raw = _link_raw_pose(link)
                node.set_position(raw[:3].tolist())
                node.set_rotation(raw[3:7].tolist())

        def tcp_report():
            rep = _to_np(unwrapped.agent.tcp_pose.raw_pose).reshape(-1, 7)[0]
            fk = _link_raw_pose(tcp_link)
            pos_err = float(np.linalg.norm(rep[:3] - fk[:3]))
            print(
                f"link_tcp world pose  p={np.round(fk[:3], 4).tolist()}  "
                f"q(wxyz)={np.round(fk[3:7], 4).tolist()}  "
                f"| agent.tcp_pose vs link_tcp FK pos_err={pos_err:.2e} m",
                flush=True,
            )

        print("=" * 72)
        print("xArm7-gripper: native coordinate frame attached to link_tcp")
        print("  arrows: x=RED  y=GREEN  z=BLUE   (link_tcp frame is the large one)")
        print("  CHECK 1: the link_tcp frame origin sits centered BETWEEN the finger tips,")
        print("           ~172 mm past the arm flange (not at the wrist/flange).")
        print("  CHECK 2: its blue (+z) axis points along the gripper approach direction.")
        print("  This is the frame the IK planner targets and the goal is measured against.")
        if not args.tcp_only:
            print("  Smaller frames mark base + link1..link7 for context.")
        print("=" * 72, flush=True)

        sync_frames()
        tcp_report()

        if args.hold > 0:
            deadline = time.monotonic() + args.hold
            while time.monotonic() < deadline:
                sync_frames()
                env.render()
                time.sleep(1.0 / 30.0)

        # Sweep each arm joint through a phase-shifted sine so link_tcp visibly
        # tracks the flange as the arm moves.
        if args.sweep:
            qpos0 = _to_np(robot.get_qpos()).reshape(-1)
            for step in range(args.sweep_steps):
                phase = 2.0 * np.pi * step / max(args.sweep_steps, 1)
                q = qpos0.copy()
                for j in range(7):
                    q[j] = qpos0[j] + 0.5 * np.sin(phase + j * 0.7)
                _set_arm_qpos(robot, q)
                sync_frames()
                env.render()
                if step % 30 == 0:
                    tcp_report()
                if args.step_delay > 0:
                    time.sleep(args.step_delay)

        # Final hold so the last configuration is inspectable.
        deadline = time.monotonic() + max(args.hold, 3.0)
        while time.monotonic() < deadline:
            sync_frames()
            env.render()
            time.sleep(1.0 / 30.0)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_gui())
