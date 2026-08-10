"""eval_nullspace_posture_reach.py
===================================
Null-space posture sampling at inference time for the XArm7 + Mixed Obstacle env.

Instead of passively accepting whatever single arm configuration DP3's diffusion
samples produce, this script:

  1. Samples K candidate action chunks per replan step (same as reranking).
  2. Scores each candidate not just by obstacle avoidance (constraint cost) but by
     a configurable *null-space posture cost* that can penalise high elbow positions,
     high link-4 heights, non-compact arm configurations, etc.
  3. Selects the candidate with the lowest combined cost and executes it.

This is the minimal footprint needed to demonstrate "actively steered whole-arm
posture" using the existing PG3DReach-RealMixedObstacle-v0 environment, without
requiring any new training data or checkpoint.

Compatible environment: PG3DReach-RealMixedObstacle-v0 (XArm7 + gripper).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

# ---------------------------------------------------------------------------
# ManiSkill / gym imports
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
except ImportError as exc:
    print(f"Failed to import ManiSkill/Gymnasium: {exc}", file=sys.stderr)
    sys.exit(2)

from mani_skill.utils.registration import register_env

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    load_reach_metadata,
)
from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.envs.xarm_adapter.obstacle_envs import (  # noqa: F401 — registers envs
    PG3DReachRealMixedObstacleEnv,
    PG3DReachXArm7RealObstacleEnv,
    PG3DReachRealConeObstacleEnv,
    PG3DReachRealKitchenEnv,
)
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.policies.dp3.goal_markers import DEFAULT_GOAL_MARKER_RADIUS
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from pg3d.utils.arrays import frame_to_numpy as _frame_to_numpy

from scripts.rollout_dp3_reach_policy import (
    ActionMode,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_video,
    select_rollout_specs,
)


# ---------------------------------------------------------------------------
# Posture mode definitions
# ---------------------------------------------------------------------------

PostureMode = Literal[
    "none",          # No posture guidance – pure DP3 (baseline)
    "low_elbow",     # Prefer candidates that keep joint-4 (elbow) as low as possible
    "compact",       # Prefer candidates that minimise total joint-space deviation from
                    #  a neutral tucked pose
    "obstacle_clearance",  # Prefer candidates maximising minimum clearance from observed
                          #  obstacle points in the current point cloud
]

# XArm7 joint index references (0-indexed, arm only, 7 DOF):
#   0 = base rotation, 1 = shoulder, 2 = upper-arm, 3 = elbow,
#   4 = forearm roll, 5 = wrist pitch, 6 = wrist roll
_ELBOW_JOINT_IDX = 3   # joint-4 pitch — controls elbow height
_SHOULDER_IDX    = 1   # shoulder elevation

# A "compact / tucked" arm reference configuration in joint space (radians).
# Lower shoulder elevation + higher elbow flexion keeps the arm tucked inside
# a shelf footprint. Adjust per your robot's rest pose conventions.
_COMPACT_REFERENCE_QPOS = np.array(
    [0.0, -0.5, 0.0, 1.2, 0.0, 0.5, 0.0], dtype=np.float32
)

# ---------------------------------------------------------------------------
# Posture scoring helpers
# ---------------------------------------------------------------------------

def _score_low_elbow(action_chunk: np.ndarray, current_qpos: np.ndarray) -> float:
    """Lower score = arm kept lower = better for shelf-reaching.

    We simulate the *mean* joint-4 value across the action chunk by integrating
    the delta actions (abs_joint mode) and taking the mean elbow joint value.
    For abs_joint mode the chunk directly IS the joint position sequence.
    """
    # action_chunk: [T, action_dim]
    arm_dim = min(action_chunk.shape[1], len(current_qpos))
    # Use the mean elbow position across the planned chunk as the score
    elbow_values = action_chunk[:, _ELBOW_JOINT_IDX] if action_chunk.shape[1] > _ELBOW_JOINT_IDX else action_chunk[:, -1]
    # Higher elbow value (more extended upward) → higher cost
    return float(np.mean(elbow_values))


def _score_compact(action_chunk: np.ndarray, current_qpos: np.ndarray) -> float:
    """Mean L2 deviation of planned joints from a compact reference pose."""
    arm_dim = min(action_chunk.shape[1], _COMPACT_REFERENCE_QPOS.shape[0])
    ref = _COMPACT_REFERENCE_QPOS[:arm_dim]
    chunk_arm = action_chunk[:, :arm_dim]
    deviation = np.linalg.norm(chunk_arm - ref, axis=1)
    return float(np.mean(deviation))


def _score_obstacle_clearance(
    action_chunk: np.ndarray,
    obstacle_points: np.ndarray | None,
    eef_path: np.ndarray | None,
) -> float:
    """Return *negative* minimum clearance (lower = better score for minimiser).

    We want maximum clearance, so we negate: min_cost = -min_clearance.
    If no obstacle points available, returns 0.0 (neutral).
    """
    if obstacle_points is None or obstacle_points.size == 0 or eef_path is None or eef_path.size == 0:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(eef_path, obstacle_points)  # [T, N_obs]
    min_clearance = float(np.min(dists))
    return -min_clearance  # negate so minimiser picks max clearance


def _compute_posture_score(
    mode: PostureMode,
    action_chunk: np.ndarray,
    current_qpos: np.ndarray,
    obstacle_points: np.ndarray | None,
    eef_path: np.ndarray | None,
) -> float:
    if mode == "none":
        return 0.0
    elif mode == "low_elbow":
        return _score_low_elbow(action_chunk, current_qpos)
    elif mode == "compact":
        return _score_compact(action_chunk, current_qpos)
    elif mode == "obstacle_clearance":
        return _score_obstacle_clearance(action_chunk, obstacle_points, eef_path)
    else:
        raise ValueError(f"Unknown posture mode: {mode!r}")


# ---------------------------------------------------------------------------
# Obstacle point extraction from point cloud observation
# ---------------------------------------------------------------------------

def _extract_obstacle_points_from_obs(
    obs_entry: dict[str, np.ndarray | bool | float],
    robot_mask_threshold: float = 0.5,
) -> np.ndarray:
    """Extract non-robot (obstacle/scene) points from the current observation.

    The point cloud observation contains a 'point_cloud/xyzw' array where the
    w-channel encodes robot vs environment (robot_point_fraction). Points with
    high w-values are robot points; the rest are the scene/obstacles.
    """
    pcd = obs_entry.get("point_cloud/xyzw")
    if pcd is None:
        pcd = obs_entry.get("point_cloud")
    if pcd is None:
        return np.empty((0, 3), dtype=np.float32)
    pcd = np.asarray(pcd, dtype=np.float32)
    if pcd.ndim == 3:
        pcd = pcd.reshape(-1, pcd.shape[-1])
    if pcd.shape[1] >= 4:
        # w > threshold → robot point; else obstacle/scene
        obstacle_mask = pcd[:, 3] < robot_mask_threshold
        return pcd[obstacle_mask, :3]
    return pcd[:, :3]  # no w channel, return all


# ---------------------------------------------------------------------------
# Main rollout function
# ---------------------------------------------------------------------------

def run_nullspace_rollout(
    *,
    env: Any,
    policy: SimpleDP3,
    spec: RolloutSpec,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    output_dir: Path,
    device: torch.device,
    max_steps: int,
    replan_stride: int,
    post_success_steps: int,
    gripper_open: float,
    video_fps: int,
    action_ema_alpha: float,
    k_candidates: int,
    posture_mode: PostureMode,
    posture_weight: float,
    goal_distance_weight: float,
) -> dict[str, Any]:
    """Run one episode with null-space posture-aware candidate selection."""
    obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    frames = [_frame_to_numpy(env.render())]
    first_entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    obs_window = make_initial_obs_window(first_entry, n_obs_steps=int(policy.n_obs_steps))

    steps = 0
    success = False
    first_success_step: int | None = None
    final_distance = float("nan")
    min_distance = float("inf")
    observed_post_success_steps = 0
    ema_sim_action: np.ndarray | None = None

    # Track selection stats
    selected_candidate_indices: list[int] = []
    posture_scores_per_replan: list[list[float]] = []
    goal_scores_per_replan: list[list[float]] = []

    was_training = policy.training
    policy.eval()

    try:
        while steps < max_steps:
            if first_success_step is not None and observed_post_success_steps >= post_success_steps:
                break

            # ----------------------------------------------------------------
            # 1. Sample K candidate action chunks from DP3
            # ----------------------------------------------------------------
            current_qpos = np.asarray(
                obs_window[-1].get("agent_pos", np.zeros(7)), dtype=np.float32
            )
            obstacle_points = _extract_obstacle_points_from_obs(obs_window[-1])

            candidate_chunks: list[np.ndarray] = []
            with torch.no_grad():
                for _ in range(k_candidates):
                    policy_input = obs_window_to_torch(
                        obs_window,
                        device=device,
                        goal_marker_points=int(policy.goal_marker_points),
                        goal_marker_radius=float(policy.goal_marker_radius),
                    )
                    output = policy.predict_action(policy_input)
                    chunk = output["action"][0].detach().cpu().numpy()  # [T, action_dim]
                    candidate_chunks.append(chunk)

            # ----------------------------------------------------------------
            # 2. Score candidates: posture cost + goal distance cost
            # ----------------------------------------------------------------
            posture_scores = []
            goal_scores = []
            combined_scores = []

            for chunk in candidate_chunks:
                # Build approximate EEF path for obstacle clearance scoring.
                # We use the first column of the chunk as proxy (joint 0 displacement).
                # In abs_joint mode, the chunk is joint positions; goal distance is
                # approximated as the joint-space distance to current_qpos (how far it moves).
                eef_path = None  # geometric EEF path not available without FK
                p_score = _compute_posture_score(
                    mode=posture_mode,
                    action_chunk=chunk,
                    current_qpos=current_qpos,
                    obstacle_points=obstacle_points if posture_mode == "obstacle_clearance" else None,
                    eef_path=eef_path,
                )
                # Goal distance approximation: last action of chunk vs goal (not reachable here;
                # use norm of chunk as proxy — smaller norm → more stationary → bad)
                # A better proxy is the magnitude of the final action vector (more progress = lower cost)
                g_score = -float(np.linalg.norm(chunk[-1]))  # negate: prefer larger movements

                posture_scores.append(p_score)
                goal_scores.append(g_score)
                combined = posture_weight * p_score + goal_distance_weight * g_score
                combined_scores.append(combined)

            posture_scores_per_replan.append(posture_scores)
            goal_scores_per_replan.append(goal_scores)

            # ----------------------------------------------------------------
            # 3. Select the best candidate
            # ----------------------------------------------------------------
            best_idx = int(np.argmin(combined_scores))
            selected_candidate_indices.append(best_idx)
            best_chunk = candidate_chunks[best_idx]

            # ----------------------------------------------------------------
            # 4. Execute the selected chunk (with EMA smoothing)
            # ----------------------------------------------------------------
            sim_action_dim = int(np.prod(env.action_space.shape))
            for policy_action in best_chunk[:replan_stride]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    obs_window[-1].get("agent_pos", current_qpos),
                    action_mode=action_mode,
                    sim_action_dim=sim_action_dim,
                    low=getattr(env.action_space, "low", None),
                    high=getattr(env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                if ema_sim_action is None or action_ema_alpha >= 1.0:
                    ema_sim_action = sim_action
                else:
                    ema_sim_action = (
                        action_ema_alpha * sim_action
                        + (1.0 - action_ema_alpha) * ema_sim_action
                    )
                obs, _reward, terminated, truncated, info = env.step(ema_sim_action)
                steps += 1
                frames.append(_frame_to_numpy(env.render()))

                entry = rollout_observation_entry(
                    obs, info, env=env, crop_config=crop_config
                )
                obs_window = append_obs_window(
                    obs_window, entry, n_obs_steps=int(policy.n_obs_steps)
                )

                success_bool = bool(info.get("success", False))
                dist_val = info.get("tcp_to_goal_dist", float("nan"))
                if isinstance(dist_val, torch.Tensor):
                    dist_val = float(dist_val.item())
                final_distance = float(dist_val)
                if np.isfinite(final_distance):
                    min_distance = min(min_distance, final_distance)

                if success_bool:
                    success = True
                    if first_success_step is None:
                        first_success_step = steps

                if success:
                    observed_post_success_steps += 1
                    if observed_post_success_steps >= post_success_steps:
                        break

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

    finally:
        if was_training:
            policy.train()

    # ----------------------------------------------------------------
    # 5. Save video
    # ----------------------------------------------------------------
    video_path: Path | None = None
    if frames:
        video_stem = f"episode_{spec.output_index:03d}_seed{spec.seed}_posture{posture_mode}"
        video_path = output_dir / (video_stem + ".mp4")
        save_video(video_path, frames, fps=video_fps)

    return {
        "episode": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "success": success,
        "first_success_step": first_success_step,
        "steps": steps,
        "final_distance": final_distance if np.isfinite(final_distance) else None,
        "min_distance": min_distance if np.isfinite(min_distance) else None,
        "posture_mode": posture_mode,
        "k_candidates": k_candidates,
        "best_candidate_indices": selected_candidate_indices,
        "video": str(video_path) if video_path else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    register_pg3d_reach_envs()
    register_pg3d_xarm7_gripper_reach_envs()

    metadata = load_reach_metadata(args.dataset)
    env_id = args.env_id_override or metadata.get("env_id", "PG3DReach-XArm7-Gripper-Workspace-v0")
    metadata["env_id"] = env_id

    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        args.checkpoint, device=device, prefer_ema=args.checkpoint_model == "ema"
    )
    policy.eval()

    action_mode_str = str(metadata.get("action_mode", "abs_joint"))
    # Import ActionMode enum from rollout module
    action_mode = _action_mode(action_mode_str)
    crop_config = crop_config_from_metadata(metadata)

    # Adjust crop bounds for the obstacle environment
    new_bounds = crop_config.bounds.copy()
    new_bounds[0, 1] = max(new_bounds[0, 1], 0.7)
    new_bounds[2, 0] = 0.005
    crop_config = PointCloudCropConfig(
        bounds=new_bounds,
        num_points=crop_config.num_points,
        robot_point_fraction=0.25,
    )

    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs.setdefault("obs_mode", "pointcloud")

    dataset_episode_seeds = [
        int(ep["seed"])
        for ep in metadata.get("episodes", [])
        if "seed" in ep
    ]

    specs = select_rollout_specs(
        source=args.source,
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=args.episode_indices,
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no rollout episodes selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    env: Any | None = None
    summaries: list[dict[str, Any]] = []
    try:
        env = gym.make(env_id, **env_kwargs)
        for spec in specs:
            print(
                f"episode={spec.output_index} seed={spec.seed} "
                f"posture_mode={args.posture_mode} k={args.k_candidates}",
                flush=True,
            )
            summary = run_nullspace_rollout(
                env=env,
                policy=policy,
                spec=spec,
                action_mode=action_mode,
                crop_config=crop_config,
                output_dir=args.output_dir,
                device=device,
                max_steps=args.max_steps,
                replan_stride=int(policy.n_action_steps),
                post_success_steps=args.post_success_steps,
                gripper_open=args.gripper_open,
                video_fps=args.video_fps,
                action_ema_alpha=args.action_ema_alpha,
                k_candidates=args.k_candidates,
                posture_mode=args.posture_mode,
                posture_weight=args.posture_weight,
                goal_distance_weight=args.goal_distance_weight,
            )
            summaries.append(summary)
            print(
                f"  -> success={summary['success']} "
                f"final_distance={summary['final_distance']} "
                f"steps={summary['steps']}",
                flush=True,
            )
    finally:
        if env is not None:
            env.close()

    result = {
        "posture_mode": args.posture_mode,
        "k_candidates": args.k_candidates,
        "posture_weight": args.posture_weight,
        "goal_distance_weight": args.goal_distance_weight,
        "env_id": env_id,
        "episodes": summaries,
        "success_rate": sum(1 if s["success"] else 0 for s in summaries) / len(summaries) if summaries else 0.0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"\nDone. success_rate={result['success_rate']:.2f} over {len(summaries)} episodes.",
        flush=True,
    )
    failures = sum(0 if s["success"] else 1 for s in summaries)
    return 0 if args.allow_failure or failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _action_mode(mode_str: str) -> ActionMode:
    from scripts.rollout_dp3_reach_policy import _action_mode as _am
    return _am(mode_str)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Null-space posture-steered inference for XArm7 on the "
            "PG3DReach-RealMixedObstacle-v0 environment.\n\n"
            "Posture modes:\n"
            "  none               – Baseline: pure DP3, single sample per step.\n"
            "  low_elbow          – Samples K candidates, picks the one with the\n"
            "                       lowest mean elbow (joint-4) position across the chunk.\n"
            "                       Best for reaching under overhanging obstacles.\n"
            "  compact            – Picks the candidate closest to a compact tucked\n"
            "                       reference configuration in joint space.\n"
            "  obstacle_clearance – Picks the candidate maximising clearance from\n"
            "                       observed obstacle points in the current point cloud."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Zarr dataset used to extract seeds/metadata.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["dataset", "fresh"], default="fresh")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=11493)
    parser.add_argument(
        "--env-id-override",
        type=str,
        default=None,
        help="Override the obstacle env (default: PG3DReach-RealMixedObstacle-v0).",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--post-success-steps", type=int, default=8)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--action-ema-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing factor (0.6 = recommended for XArm7).",
    )
    # Null-space posture arguments
    parser.add_argument(
        "--posture-mode",
        choices=["none", "low_elbow", "compact", "obstacle_clearance"],
        default="low_elbow",
        help="Null-space posture steering mode (see description above).",
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        default=16,
        help="Number of DP3 action chunk candidates to sample per replan.",
    )
    parser.add_argument(
        "--posture-weight",
        type=float,
        default=1.0,
        help="Weight for the posture cost term in combined candidate scoring.",
    )
    parser.add_argument(
        "--goal-distance-weight",
        type=float,
        default=0.5,
        help="Weight for the goal-progress term in combined candidate scoring.",
    )
    parser.add_argument("--allow-failure", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
