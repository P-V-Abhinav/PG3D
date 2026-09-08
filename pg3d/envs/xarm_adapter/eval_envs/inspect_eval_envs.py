"""Open the SAPIEN viewer on each eval env, one at a time, and look at it.

    python -m pg3d.envs.xarm_adapter.eval_envs.inspect_eval_envs
    python -m pg3d.envs.xarm_adapter.eval_envs.inspect_eval_envs --skip-cluttered
    python -m pg3d.envs.xarm_adapter.eval_envs.inspect_eval_envs \
        --env-ids PG3DReach-Eval-Obs-PP-v3 --show-workspace --settle-steps 60

Needs a display (X11/Wayland) and a working SAPIEN renderer. The env is built,
reset to its frozen state, and held there while a viewer window is open; the
terminal prints everything the env froze -- start TCP and the error against it,
the policy goal, the object/obstacle layout, the active camera, and the
clearance from every obstacle to the start, goal and target object -- so the
numbers can be checked against what is on screen.

Keys, while the viewer window has focus
---------------------------------------
    n / right     next env
    p / left      previous env
    r             reset the current env (re-randomises only the camera jitter)
    m             toggle the start/goal marker balls (viewer only -- they are
                  hidden from every observation either way)
    o             print this env's frozen numbers again
    q / esc       quit

SAPIEN's own control window (top-left of the viewer) is still there: pause,
free-fly the camera with WASD + right-drag, click an actor to inspect it, and
pick a specific sensor camera from its camera list to see exactly what that
camera sees.

Windows are opened per env: advancing closes the current window and opens the
next one, because ManiSkill binds a viewer to one scene.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

from pg3d.envs.xarm_adapter.eval_envs import all_eval_env_ids, register_pg3d_eval_envs
from pg3d.envs.xarm_adapter.eval_envs.eval_config import (
    OBSTACLE_HALF_SIZES,
    START_TCP,
    START_TCP_TOLERANCE_M,
)

#: Keys that advance/leave, mapped to the action name the loop returns.
_KEYS = {
    "n": "next",
    "right": "next",
    "p": "prev",
    "left": "prev",
    "r": "reset",
    "m": "markers",
    "o": "info",
    "q": "quit",
    "esc": "quit",
}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _box_surface_distance(centre: Any, point: Any) -> float:
    """Distance from an obstacle box's surface to a point (0 if inside)."""
    delta = np.abs(np.asarray(point, dtype=float) - np.asarray(centre, dtype=float))
    outside = np.maximum(delta - np.asarray(OBSTACLE_HALF_SIZES, dtype=float), 0.0)
    return float(np.linalg.norm(outside))


def _print_env_info(env: Any) -> None:
    unwrapped = env.unwrapped
    spec = unwrapped.pg3d_eval_spec
    achieved = _numpy(unwrapped.agent.tcp_pose.p).reshape(-1, 3)[0]
    start_error = float(np.linalg.norm(achieved - np.asarray(spec.start_tcp, dtype=float)))
    flag = "OK" if start_error <= START_TCP_TOLERANCE_M else "OUT OF TOLERANCE"

    print(f"\n=== {spec.env_id}   tasks {', '.join(spec.task_ids) or '-'}")
    print(f"  start TCP   frozen {tuple(round(v, 4) for v in spec.start_tcp)}")
    print(f"              actual {tuple(round(float(v), 4) for v in achieved)}"
          f"   error {start_error:.4f} m  [{flag}]")
    print(f"  policy goal        {tuple(round(v, 4) for v in spec.goal_position)}"
          f"   (success radius {spec.goal_threshold_m} m)")
    print(f"  goal marker        {spec.marker_points} pts, r={spec.marker_radius} m,"
          f" {spec.marker_shape} -- appended to the point cloud, never rendered")
    if spec.target_object:
        print(f"  target object      {spec.target_object} at "
              f"{tuple(round(v, 4) for v in spec.target_object_position or ())}")
    for name, position in spec.clutter.items():
        print(f"    clutter          {name} at {tuple(round(v, 4) for v in position)}")

    if spec.obstacles:
        print(f"  obstacles          {len(spec.obstacles)} bars, "
              f"half-sizes {OBSTACLE_HALF_SIZES}")
        probes = {"start": spec.start_tcp, "goal": spec.goal_position}
        if spec.target_object_position:
            probes["target"] = spec.target_object_position
        for name, position in spec.obstacles.items():
            gaps = ", ".join(
                f"{label} {_box_surface_distance(position, probe):.3f} m"
                for label, probe in probes.items()
            )
            print(f"    {name:12s} at {tuple(round(v, 4) for v in position)}   clearance: {gaps}")

    cameras = ", ".join(sorted(unwrapped._sensors.keys()))
    print(f"  sensor cameras     {cameras}")
    print(f"  episode cap        {spec.max_episode_steps} steps")


def _inspect_one(
    env_id: str,
    args: argparse.Namespace,
) -> str:
    """Show one env in the viewer. Returns the action the user asked for."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(
        env_id,
        obs_mode="pointcloud",
        num_envs=1,
        render_mode="human",
        show_workspace=args.show_workspace,
        strict_start=not args.no_strict_start,
    )
    try:
        env.reset(seed=args.seed, options={"reconfigure": True})
        unwrapped = env.unwrapped
        _print_env_info(env)
        if args.marker_report:
            print(f"  marker exclusion   {unwrapped.marker_leak_report()}")

        if args.settle_steps:
            # Hold the arm still and let physics settle, so a badly seated
            # object shows itself instead of looking fine in a frozen frame.
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            qpos = _numpy(unwrapped.agent.robot.get_qpos()).reshape(-1)
            action[: min(7, action.shape[0])] = qpos[:7]
            for _ in range(int(args.settle_steps)):
                env.step(action)

        markers_visible = True
        pressed: set[str] = set()
        deadline = None if args.max_seconds is None else time.time() + args.max_seconds
        print("  [n]ext  [p]rev  [r]eset  [m]arkers  [o]info  [q]uit")

        while True:
            viewer = env.render()
            if viewer is None or viewer.closed:
                return "quit"
            window = viewer.window
            down = {key for key in _KEYS if window.key_down(key)}
            for key in down - pressed:  # edge-triggered, so a held key fires once
                action_name = _KEYS[key]
                if action_name == "markers":
                    markers_visible = not markers_visible
                    unwrapped.set_marker_visibility(markers_visible)
                    print(f"  markers {'shown' if markers_visible else 'hidden'} "
                          "(viewer only; observations always hide them)")
                elif action_name == "info":
                    _print_env_info(env)
                elif action_name == "reset":
                    env.reset(seed=args.seed)
                    print("  reset")
                else:
                    return action_name
            pressed = down
            if deadline is not None and time.time() > deadline:
                print(f"  --max-seconds {args.max_seconds} reached, moving on")
                return "next"
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-ids", nargs="+", default=None, help="default: all 25")
    parser.add_argument("--skip-cluttered", action="store_true",
                        help="skip the YCB envs (before the YCB assets are downloaded)")
    parser.add_argument("--show-workspace", action="store_true",
                        help="draw the green workspace bounding box")
    parser.add_argument("--settle-steps", type=int, default=0,
                        help="hold the arm still for N physics steps before viewing")
    parser.add_argument("--marker-report", action="store_true",
                        help="also run the differential marker-exclusion check per env")
    parser.add_argument("--no-strict-start", action="store_true",
                        help="warn instead of raising if the start TCP misses")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="auto-advance after N seconds (for unattended checks)")
    args = parser.parse_args(argv)

    register_pg3d_eval_envs()
    env_ids = list(args.env_ids or all_eval_env_ids())
    if args.skip_cluttered:
        env_ids = [env_id for env_id in env_ids if "Cluttered" not in env_id]
    if not env_ids:
        print("no envs selected", file=sys.stderr)
        return 2

    print(f"frozen start TCP: {tuple(round(v, 4) for v in START_TCP)}")
    print(f"inspecting {len(env_ids)} envs")

    index = 0
    while 0 <= index < len(env_ids):
        env_id = env_ids[index]
        try:
            action = _inspect_one(env_id, args)
        except Exception as exc:
            print(f"\n!! {env_id}: {type(exc).__name__}: {exc}\n", file=sys.stderr)
            action = "next"
        if action == "quit":
            print("bye")
            return 0
        index += -1 if action == "prev" else 1
        if index < 0:
            index = 0
    print("end of list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
