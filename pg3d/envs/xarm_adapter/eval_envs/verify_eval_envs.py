"""Smoke-verify the frozen eval suite on a machine that has the simulator.

Run this once on the box that will produce the results, before any eval run:

    python -m pg3d.envs.xarm_adapter.eval_envs.verify_eval_envs
    python -m pg3d.envs.xarm_adapter.eval_envs.verify_eval_envs --env-ids PG3DReach-Eval-Reach-v1
    python -m pg3d.envs.xarm_adapter.eval_envs.verify_eval_envs --skip-cluttered --json report.json

For each env it builds the env, resets it, and reports:

* the achieved start TCP and its error against the frozen ``START_TCP`` -- the
  number that decides whether the frozen obstacle/object clearances hold;
* the goal the policy will be conditioned on, and the goal-marker contract;
* which sensor cameras are active -- there must be exactly one, ``base_camera``,
  the calibrated eye-on-base view the checkpoint was trained on;
* how many camera points survive the crop, and how many points hiding the
  start/goal markers removes from the observation -- which must be non-zero, and
  is the decisive test that they are excluded;
* the frozen object/obstacle positions the env actually applied.

Exit code is non-zero if any env fails, so it can gate a run in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

import numpy as np

from pg3d.envs.xarm_adapter.eval_envs import all_eval_env_ids, register_pg3d_eval_envs
from pg3d.envs.xarm_adapter.eval_envs.eval_config import (
    GOAL_MARKER_RADIUS,
    START_TCP,
    START_TCP_TOLERANCE_M,
    workspace_report,
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _check_env(env_id: str, *, strict_start: bool) -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(env_id, obs_mode="pointcloud", num_envs=1, strict_start=strict_start)
    try:
        env.reset(seed=0, options={"reconfigure": True})
        unwrapped = env.unwrapped
        # Unflattened, so the raw point cloud is inspectable for the leak checks.
        obs = unwrapped.get_obs(unwrapped.get_info(), unflattened=True)
        spec = unwrapped.pg3d_eval_spec

        achieved = _numpy(unwrapped.agent.tcp_pose.p).reshape(-1, 3)[0].astype(float)
        start_error = float(np.linalg.norm(achieved - np.asarray(spec.start_tcp, dtype=float)))

        xyzw = _numpy(obs["pointcloud"]["xyzw"]).reshape(-1, 4)
        points = xyzw[xyzw[:, 3] > 0][:, :3]
        goal = np.asarray(spec.goal_position, dtype=float)
        # Marker exclusion, tested differentially: render once with the markers
        # hidden (the real path) and once with them present, and require that
        # hiding actually removed points. An absolute "points near the goal"
        # count cannot decide this -- a place target sits 3.5 cm above the
        # table, so the table legitimately fills a 5.5 cm ball around it.
        try:
            leak = unwrapped.assert_markers_excluded()
            marker_leak = False
        except AssertionError as exc:
            leak = unwrapped.marker_leak_report()
            marker_leak = True
            print(f"  {exc}")
        goal_ball = int(
            np.count_nonzero(np.linalg.norm(points - goal, axis=1) <= GOAL_MARKER_RADIUS)
        )
        return {
            "env_id": env_id,
            "ok": start_error <= START_TCP_TOLERANCE_M and not marker_leak,
            "marker_leak": marker_leak,
            "marker_exclusion": leak,
            "start_tcp_frozen": list(spec.start_tcp),
            "start_tcp_achieved": achieved.tolist(),
            "start_tcp_error_m": start_error,
            "start_tcp_tolerance_m": START_TCP_TOLERANCE_M,
            "goal_position": list(spec.goal_position),
            "marker": {
                "num_points": spec.marker_points,
                "radius": spec.marker_radius,
                "shape": spec.marker_shape,
            },
            "valid_points": int(points.shape[0]),
            "points_in_goal_ball": goal_ball,
            "target_object": spec.target_object,
            "target_object_position": (
                list(spec.target_object_position) if spec.target_object_position else None
            ),
            "obstacles": {name: list(pos) for name, pos in spec.obstacles.items()},
            "clutter": {name: list(pos) for name, pos in spec.clutter.items()},
            "max_episode_steps": spec.max_episode_steps,
            "sensor_cameras": sorted(unwrapped._sensors.keys()),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-ids", nargs="+", default=None, help="default: all 25")
    parser.add_argument(
        "--skip-cluttered",
        action="store_true",
        help="skip the YCB envs (useful before the YCB assets are downloaded)",
    )
    parser.add_argument(
        "--no-strict-start",
        action="store_true",
        help="warn instead of raising when the start TCP misses, to see every error at once",
    )
    parser.add_argument("--json", type=str, default=None, help="write the full report here")
    args = parser.parse_args(argv)

    register_pg3d_eval_envs()
    env_ids = list(args.env_ids or all_eval_env_ids())
    if args.skip_cluttered:
        env_ids = [env_id for env_id in env_ids if "Cluttered" not in env_id]

    print(f"frozen start TCP: {tuple(round(v, 4) for v in START_TCP)}")
    report: dict[str, Any] = {"frame": workspace_report(), "envs": []}
    failures = 0
    for env_id in env_ids:
        try:
            row = _check_env(env_id, strict_start=not args.no_strict_start)
        except Exception as exc:
            failures += 1
            row = {"env_id": env_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        else:
            failures += 0 if row["ok"] else 1
        report["envs"].append(row)
        status = "ok " if row.get("ok") else "FAIL"
        if "error" in row:
            print(f"{status} {env_id}: {row['error']}")
        else:
            print(
                f"{status} {env_id}: start_err={row['start_tcp_error_m']:.4f}m "
                f"goal={tuple(round(v, 3) for v in row['goal_position'])} "
                f"points={row['valid_points']} "
                f"marker_leak={'YES' if row['marker_leak'] else 'no'} "
                f"marker_pts_removed={row['marker_exclusion']['removed']} "
                f"cams={len(row['sensor_cameras'])}"
            )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    print(f"{len(env_ids) - failures}/{len(env_ids)} envs ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
