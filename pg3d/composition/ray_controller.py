"""
ray_controller.py
=================
Parallel candidate scoring using Ray actor workers.

Each GhostRenderWorker owns one persistent SAPIEN ghost env (xArm7 + obstacle)
in its own OS process, giving it an isolated OpenGL/EGL context.

The main entry point is `parallel_sample_and_score`, which is a drop-in
replacement for the serial for-loop inside BaseController._sample_and_score.
All rejection/reranking logic, k-schedule values, and scoring semantics remain
100% unchanged — we only change *where* the 8 SAPIEN renders happen.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Thread-count caps MUST be set before any numpy/torch import inside workers.
# Each worker is a fresh Ray subprocess, so these os.environ calls take effect.
# --------------------------------------------------------------------------
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from typing import Any

import numpy as np
import ray
from ray.util.actor_pool import ActorPool
from scipy.spatial.distance import cdist as scipy_cdist

from pg3d.world_model import ActionChunk, ImaginedRollout
from pg3d.composition.types import CandidateDiagnostics, ScoreWeights
from pg3d.composition.scoring import (
    consensus_deviations,
    goal_distance,
    optional_policy_surrogate,
    trajectory_smoothness,
)


# ---------------------------------------------------------------------------
# Ray remote actor — one per CPU core, owns one SAPIEN ghost env
# ---------------------------------------------------------------------------
@ray.remote(num_cpus=1, num_gpus=0.05)
class GhostRenderWorker:
    """
    Persistent Ray actor that wraps one SAPIEN ghost env.

    Lifecycle
    ---------
    1. __init__: create ghost env + geometry provider inside this process.
    2. reset_episode: called once per episode to mirror seed + obstacle pose.
    3. imagine_and_score: called once per candidate per step; returns scored dict.
    """

    def __init__(
        self,
        env_id: str,
        env_kwargs: dict[str, Any],
        crop_bounds: list | None,
        task_name: str,
    ) -> None:
        # Enforce single-threaded numpy/torch inside this worker process.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers base ManiSkill envs

        # Import xArm7 base envs (triggers their @register_env decorators)
        import pg3d.envs.xarm_adapter.reach_env  # noqa: F401

        # Import obstacle envs (triggers their @register_env decorators).
        # We do a try/except in case the env_id is the base xArm7 env
        # rather than an obstacle variant.
        try:
            from pg3d.envs.xarm_adapter.obstacle_envs import (  # noqa: F401
                PG3DReachXArm7RealObstacleEnv,
                PG3DReachRealConeObstacleEnv,
                PG3DReachRealMixedObstacleEnv,
                PG3DReachRealKitchenEnv,
            )
        except Exception:
            pass

        from pg3d.envs.maniskill_adapter.geometry import (
            ManiSkillGhostPandaGeometryProvider,
        )
        from pg3d.world_model.core import GeometricWorldModel

        # Build ghost env with render_mode=None — SAPIEN uses headless EGL,
        # one independent OpenGL context per process.
        ghost_kwargs = dict(env_kwargs)
        ghost_kwargs.pop("render_mode", None)
        ghost_kwargs["num_envs"] = 1
        ghost_kwargs["obs_mode"] = "pointcloud"

        self.env = gym.make(env_id, **ghost_kwargs)
        crop = (
            np.array(crop_bounds, dtype=np.float32)
            if crop_bounds is not None
            else None
        )
        self.provider = ManiSkillGhostPandaGeometryProvider(
            self.env,
            task_name=task_name,
            crop_bounds=crop,
        )
        self.world_model = GeometricWorldModel(self.provider)

    # ------------------------------------------------------------------
    def reset_episode(
        self,
        seed: int,
        obstacle_center_xyz: list[float] | None,
    ) -> bool:
        """
        Reset ghost env to match the live sim env for this episode.
        Mirrors provider.reset() + obstacle teleport from _constraints_for_episode.
        """
        import torch
        from mani_skill.utils.structs import Pose

        self.env.reset(seed=seed, options={"reconfigure": True})
        self.provider._cache = None

        # Teleport obstacle to the same XYZ as the live sim
        if obstacle_center_xyz is not None:
            unwrapped = getattr(self.env, "unwrapped", self.env)
            if hasattr(unwrapped, "obstacle"):
                pos = torch.tensor(
                    obstacle_center_xyz, dtype=torch.float32
                ).unsqueeze(0)
                unwrapped.obstacle.set_pose(Pose.create_from_pq(pos))
                self.provider._cache = None  # invalidate any cached snapshot
        return True

    # ------------------------------------------------------------------
    def imagine_and_score(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Run the 8-step SAPIEN render loop for one candidate, compute the
        collision cost + full scoring, and return a plain serialisable dict.

        task keys
        ---------
        joint_positions, point_cloud, robot_mask, tcp_pose, target_position :
            numpy arrays — reconstruct the Observation inside the worker
        actions, action_mode, dt : reconstruct the ActionChunk
        obstacle_points, margin, weight : collision constraint
        max_robot_points : robot point budget (from match_current_robot_points)
        attempted_k, index, consensus_dev, surrogate, score_weights : scoring
        """
        from pg3d.envs.maniskill_adapter.types import (
            Observation,
            RobotState,
            SimGroundTruth,
        )
        from pg3d.world_model.types import ActionMode

        # ---- Set robot point budget ----------------------------------------
        if task.get("max_robot_points") is not None:
            self.provider.max_robot_points = int(task["max_robot_points"])
            self.provider._cache = None

        # ---- Reconstruct Observation ----------------------------------------
        robot_state = RobotState(
            joint_positions=np.asarray(task["joint_positions"], dtype=np.float32),
            joint_velocities=None,
            tcp_pose=np.asarray(task["tcp_pose"], dtype=np.float32),
            gripper_open=None,
            metadata={},
        )
        sim_gt = SimGroundTruth(
            task_name=task.get("task_name", "unknown"),
            target_position=np.asarray(task["target_position"], dtype=np.float32),
            success=None,
            metadata={},
        )
        obs = Observation(
            point_cloud=np.asarray(task["point_cloud"], dtype=np.float32),
            point_features={},
            robot_state=robot_state,
            robot_mask=np.asarray(task["robot_mask"], dtype=bool),
            object_masks={},
            sim_gt=sim_gt,
            metadata={},
        )

        # ---- Reconstruct ActionChunk ----------------------------------------
        chunk = ActionChunk(
            actions=np.asarray(task["actions"], dtype=np.float32),
            action_mode=task["action_mode"],
            dt=float(task["dt"]),
            metadata={"candidate_index": task["index"]},
        )

        # ---- Run imagine (8 SAPIEN renders) ---------------------------------
        rollout = self.world_model.imagine(obs, chunk)

        # ---- Collision cost (cdist) ------------------------------------------
        obstacle_points = np.asarray(task["obstacle_points"], dtype=np.float32)
        margin = float(task["margin"])
        weight = float(task["weight"])

        if obstacle_points.size == 0:
            max_v, mean_v, collision_penalty = 0.0, 0.0, 0.0
            feasible = True
        else:
            clouds = [c for c in rollout.robot_point_clouds if c.size]
            if clouds:
                all_robot_pts = np.concatenate(clouds, axis=0)
                dists = scipy_cdist(all_robot_pts, obstacle_points)
                min_dists = np.min(dists, axis=1)
                violations = np.maximum(margin - min_dists, 0.0)
                max_v = float(np.max(violations))
                mean_v = float(np.mean(violations))
                collision_penalty = weight * (max_v + 0.5 * mean_v)
                feasible = max_v <= 1e-6
            else:
                max_v, mean_v, collision_penalty = 0.0, 0.0, 0.0
                feasible = True

        constraint_name = task.get("constraint_name", "pointcloud_obstacle_avoid_region")
        constraint_costs: dict[str, float] = {
            constraint_name: collision_penalty,
            f"{constraint_name}/max_violation": max_v,
            f"{constraint_name}/mean_violation": mean_v,
        }
        constraint_satisfied: dict[str, bool] = {
            f"0:{constraint_name}": feasible,
        }

        # ---- Other scoring terms --------------------------------------------
        target_pos = np.asarray(task["target_position"], dtype=np.float32)
        dist = goal_distance(rollout, target_pos)
        smoothness = trajectory_smoothness(rollout, order=2)
        consensus_dev = float(task["consensus_dev"])
        surrogate = task.get("surrogate")
        sw: dict[str, float] = task["score_weights"]

        total_score = (
            sw["constraint"] * collision_penalty
            + sw["goal_distance"] * (0.0 if dist is None else dist)
            + sw["smoothness"] * smoothness
            + sw["consensus"] * consensus_dev
            + sw["policy_surrogate"] * (0.0 if surrogate is None else surrogate)
            + sw.get("directional", 0.0) * 0.0  # directional=0 here (no sign set)
        )

        # ---- Pack rollout data for reconstruction in main process -----------
        return {
            # scoring results
            "index": int(task["index"]),
            "attempted_k": int(task["attempted_k"]),
            "feasible": bool(feasible),
            "constraint_costs": constraint_costs,
            "constraint_satisfied": constraint_satisfied,
            "goal_distance": dist,
            "constraint_penalty": collision_penalty,
            "smoothness": smoothness,
            "consensus_deviation": consensus_dev,
            "policy_surrogate": surrogate,
            "total_score": float(total_score),
            "directional": 0.0,
            # rollout numpy arrays (for ImaginedRollout reconstruction)
            "q": np.asarray(rollout.q, dtype=np.float32),
            "eef_path": np.asarray(rollout.eef_path, dtype=np.float32),
            "robot_point_clouds": [
                np.asarray(c, dtype=np.float32) for c in rollout.robot_point_clouds
            ],
            "scene_point_clouds": [
                np.asarray(c, dtype=np.float32) for c in rollout.scene_point_clouds
            ],
            "robot_masks": [
                np.asarray(m, dtype=bool) for m in rollout.robot_masks
            ],
            "eef_orientations": (
                np.asarray(rollout.eef_orientations, dtype=np.float32)
                if rollout.eef_orientations is not None
                else None
            ),
            "rollout_metadata": dict(rollout.metadata),
        }


# ---------------------------------------------------------------------------
# Build a pool of Ray actors
# ---------------------------------------------------------------------------
def build_ray_actor_pool(
    n_workers: int,
    env_id: str,
    env_kwargs: dict[str, Any],
    crop_bounds: np.ndarray | None,
    task_name: str,
) -> ActorPool:
    """
    Initialise Ray (if not already running) and create n_workers actor instances.

    Parameters
    ----------
    n_workers:   number of Ray actors (one per CPU core you want to use)
    env_id:      the resolved / overridden env id (e.g. PG3DReach-RealConeObstacle-v0)
    env_kwargs:  env kwargs dict from the zarr metadata (obs_mode, num_envs stripped/set)
    crop_bounds: shape (3,2) numpy array or None
    task_name:   task name string forwarded to ManiSkillGhostPandaGeometryProvider
    """
    ray.init(
        num_cpus=n_workers + 2,   # +2 for main process + dashboard
        ignore_reinit_error=True,
    )
    crop_list = crop_bounds.tolist() if crop_bounds is not None else None
    workers = [
        GhostRenderWorker.remote(env_id, env_kwargs, crop_list, task_name)
        for _ in range(n_workers)
    ]
    return ActorPool(workers)


# ---------------------------------------------------------------------------
# Drop-in replacement for BaseController._sample_and_score
# ---------------------------------------------------------------------------
def parallel_sample_and_score(
    actor_pool: ActorPool,
    controller: Any,           # BaseController instance
    controller_input: Any,     # ControllerInput
    *,
    attempted_k: int,
    start_index: int,
    rng: np.random.Generator | None,
    obstacle_points: np.ndarray,
    collision_margin: float,
    collision_weight: float,
    collision_constraint_name: str,
    max_robot_points: int | None,
    task_name: str,
) -> list[CandidateDiagnostics]:
    """
    Parallel equivalent of BaseController._sample_and_score.

    1.  Sample k action chunks from the policy (GPU, main process).
    2.  Fan-out to Ray actor pool: each worker runs imagine() + score().
    3.  Collect results, reconstruct CandidateDiagnostics list.
    4.  Return identical structure to the serial version.
    """
    policy_input = controller_input.input_for_policy()
    chunks = controller.policy.sample_action_chunks(
        policy_input, k=attempted_k, rng=rng
    )
    if not chunks:
        return []

    surrogates = optional_policy_surrogate(controller.policy, policy_input, chunks)
    consensus = consensus_deviations(chunks)
    obs = controller_input.observation
    sw = controller.score_weights

    score_weights_dict = {
        "constraint":     sw.constraint,
        "goal_distance":  sw.goal_distance,
        "smoothness":     sw.smoothness,
        "consensus":      sw.consensus,
        "policy_surrogate": sw.policy_surrogate,
        "directional":    sw.directional,
    }

    # Build one task dict per candidate.  All values must be cloudpickle-safe
    # (numpy arrays, plain Python scalars/dicts/lists — all fine).
    tasks = []
    for local_idx, chunk in enumerate(chunks):
        tasks.append(
            {
                # observation fields
                "joint_positions": np.asarray(
                    obs.robot_state.joint_positions, dtype=np.float32
                ),
                "point_cloud": np.asarray(obs.point_cloud, dtype=np.float32),
                "robot_mask": np.asarray(obs.robot_mask, dtype=bool),
                "tcp_pose": (
                    np.asarray(obs.robot_state.tcp_pose, dtype=np.float32)
                    if obs.robot_state.tcp_pose is not None
                    else np.zeros(7, dtype=np.float32)
                ),
                "target_position": (
                    np.asarray(obs.sim_gt.target_position, dtype=np.float32)
                    if obs.sim_gt is not None and obs.sim_gt.target_position is not None
                    else np.zeros(3, dtype=np.float32)
                ),
                "task_name": task_name,
                # action chunk
                "actions": np.asarray(chunk.actions, dtype=np.float32),
                "action_mode": chunk.action_mode,
                "dt": float(chunk.dt),
                # collision constraint
                "obstacle_points": obstacle_points,
                "margin": collision_margin,
                "weight": collision_weight,
                "constraint_name": collision_constraint_name,
                # point budget
                "max_robot_points": max_robot_points,
                # scoring metadata
                "attempted_k": attempted_k,
                "index": start_index + local_idx,
                "consensus_dev": float(consensus[local_idx]),
                "surrogate": (
                    float(surrogates[local_idx])
                    if surrogates[local_idx] is not None
                    else None
                ),
                "score_weights": score_weights_dict,
            }
        )

    # Dispatch all candidates to the pool in parallel.
    # map_unordered returns results as they complete; we re-sort by index below.
    results: list[dict[str, Any]] = list(
        actor_pool.map_unordered(lambda w, t: w.imagine_and_score.remote(t), tasks)
    )

    # Re-sort by original candidate index (map_unordered is non-deterministic order).
    results.sort(key=lambda r: r["index"])

    # Reconstruct CandidateDiagnostics — identical structure to the serial path.
    diagnostics: list[CandidateDiagnostics] = []
    for result, chunk in zip(results, chunks):
        rollout = ImaginedRollout(
            q=result["q"],
            eef_path=result["eef_path"],
            robot_point_clouds=result["robot_point_clouds"],
            scene_point_clouds=result["scene_point_clouds"],
            robot_masks=result["robot_masks"],
            action_chunk=chunk,
            metadata=result["rollout_metadata"],
            eef_orientations=result.get("eef_orientations"),
        )
        diag = CandidateDiagnostics(
            index=result["index"],
            attempted_k=result["attempted_k"],
            action_chunk=chunk,
            rollout=rollout,
            constraint_costs=result["constraint_costs"],
            constraint_satisfied=result["constraint_satisfied"],
            feasible=result["feasible"],
            goal_distance=result["goal_distance"],
            constraint_penalty=result["constraint_penalty"],
            smoothness=result["smoothness"],
            consensus_deviation=result["consensus_deviation"],
            policy_surrogate=result["policy_surrogate"],
            total_score=result["total_score"],
            directional=result.get("directional", 0.0),
        )
        diagnostics.append(diag)

    return diagnostics
