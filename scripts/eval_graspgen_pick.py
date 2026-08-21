from __future__ import annotations

"""eval_graspgen_pick.py — Phase 0: GraspGen-steered grasp approach.

Architecture
============
1. Before the episode loop: load GraspGenSampler ONCE from --graspgen-config.
2. Per episode reset:
   a. Get scene point cloud from ManiSkill cameras (one zero-action step to
      force the renderer to capture the post-init scene).
   b. **Object pose**: For jstbanana-v0 (and any env with `cheezit` actor),
      the grasp target position is read DIRECTLY from the actor's world-frame
      pose — NOT from entry["target_position"] (goal_site). This avoids the
      convoluted goal→object indirection and works regardless of where the
      object is placed in the workspace.
   c. Crop the raw point cloud to a sphere around the actor centroid.
   d. Run GraspGen (diffusion + OBB MoE) on the object crop.
   e. Pick the best-scoring grasp candidate.
   f. Apply --graspgen-z-offset to adjust for gripper geometry differences.
   g. Wrap as a CartesianPoseConstraint.
3. Reranking loop (UNCHANGED from pose-steering eval):
   * Sample k action chunks from the reach checkpoint.
   * Score each by CartesianPoseConstraint cost + goal_distance + smoothness.
   * Execute the chunk whose imagined EEF path gets closest to the grasp pose.
4. Debug visualisations (--graspgen-viser / --rerun):
   * Viser: launches a local 3-D server showing the object point cloud and
     ALL GraspGen candidate grasps as pitchfork frames.
   * Rerun: logs the best grasp AND all candidates as pitchfork line strips
     (approach arm + two finger-span lines) inside the existing timeline.

Gripper masking
===============
The reach checkpoint is 7-DOF (arm only). The XArm7 action space is 8-DOF
(7 arm + 1 gripper finger). policy_action_to_sim_action() already pads with
gripper_open=args.gripper_open (default open = 0.04 m). No retraining needed.

Confirmed GraspGen output schema (from server probe):
  grasps : (N, 4, 4) float32 SE(3) homogeneous matrices, world frame.
           Position = T[:3, 3], Rotation = T[:3, :3].
  scores : (N,)      float32 discriminator confidences in [0, 1].

Z-offset
========
The project's xarm7_robotiq.urdf uses the exact Robotiq 2F-140 geometry that
GraspGen was trained for → Z_OFFSET = 0.0 by default. If you ever swap the
physical gripper, measure the new fingertip depth from the URDF and pass
--graspgen-z-offset.

Phase 1 (gripper close) is a separate step. This script only steers the arm.
"""

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import zarr

# ---------------------------------------------------------------------------
# GraspGen imports — guarded so the script is importable without GraspGen.
# The actual import path was confirmed by running grasp_server API probe on
# the server:  grasp_gen.grasp_server  and  grasp_gen.samplers.graspmoe
# ---------------------------------------------------------------------------
try:
    import logging as _logging_graspgen
    # Suppress the noisy spconv FutureWarning before importing grasp_gen
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.samplers.graspmoe import run_graspmoe

    _GRASPGEN_AVAILABLE = True
except ImportError as _graspgen_err:
    _GRASPGEN_AVAILABLE = False
    GraspGenSampler = None       # type: ignore[assignment,misc]
    load_grasp_cfg = None        # type: ignore[assignment]
    run_graspmoe = None          # type: ignore[assignment]

# ---------------------------------------------------------------------------
# PG3D / ManiSkill imports  (identical to eval_pointcloud_pose_steering_reach)
# ---------------------------------------------------------------------------
from pg3d.composition import (
    CandidateDiagnostics,
    ControllerInput,
    ControllerResult,
    RejectionController,
    RerankingController,
    ScoreWeights,
)
from pg3d.composition.scoring import (
    consensus_deviations,
    goal_distance,
    primary_constraint_penalty,
    trajectory_smoothness,
)
from pg3d.constraints import (
    AvoidProjection,
    AvoidRegion,
    BoxRegion,
    CartesianPoseConstraint,
    JointPostureConstraint,
    RectRegion2D,
    SphereRegion,
)
from pg3d.constraints.core import SceneContext
from pg3d.envs.maniskill_adapter import (
    ManiSkillGhostPandaGeometryProvider,
    register_pg3d_reach_envs,
)
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    load_reach_metadata,
)
from pg3d.envs.maniskill_adapter.reach_env import PG3DReachEnv
from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    TimingRecorder,
    candidate_feasibility_fraction,
    cartesian_pose_step_metrics,
    concatenate_rollouts,
    direct_path_avoid_region,
    episode_metric_row,
    load_episode_constraints,
    progress_series,
    save_episode_constraints,
    scene_context_for_constraints,
    select_artifact_episode_indices,
    should_emit_episode_artifact,
    summarize_metrics,
    validate_planning_horizons,
)
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.checkpoint import (
    latest_reach_checkpoint,
    load_reach_policy_from_checkpoint,
)
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_RADIUS,
    insert_goal_marker_points,
)
from pg3d.utils.arrays import bool_any as _bool_any
from pg3d.utils.arrays import bool_info as _bool_info
from pg3d.utils.arrays import frame_to_numpy as _frame_to_numpy
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from pg3d.world_model import ActionChunk, GeometricWorldModel, ImaginedRollout
from pg3d.world_model.chunks import interpret_joint_chunk
from pg3d.world_model.compositor import compose_robot_cloud, static_scene_from_robot_mask
from scipy.spatial.distance import cdist

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

# xArm7 obstacle env classes register their env IDs on import.
from pg3d.envs.xarm_adapter.obstacle_envs import (
    PG3DReachRealConeObstacleEnv,   # noqa: F401
    PG3DReachRealKitchenEnv,        # noqa: F401
    PG3DReachRealMixedObstacleEnv,  # noqa: F401
    PG3DReachXArm7RealObstacleEnv,  # noqa: F401
    _create_cone_obj,
)

from scripts.compare_world_model_rollout import (
    entry_to_world_model_observation,
    world_model_entry_from_rollout_step,
)
from scripts.eval_reach_checkpoint_unique_seeds import (
    _apply_zarr_initial_entry,
    _reset_to_zarr_episode,
    _zarr_episode_context,
)
from scripts.rollout_dp3_reach_policy import (
    ActionMode,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_rerun_timeline as _save_rerun_timeline_base,
    save_video,
    select_rollout_specs,
)


# save_rerun_timeline is re-exported so any code that imports it from this
# module still works.  GraspGen pitchfork logging is done explicitly in main()
# after run_eval_episode returns, writing a companion .pitchforks.rrd file.
# Store the current episode's graspgen data here before calling run_eval_episode
CURRENT_RERUN_DATA = None

def save_rerun_timeline(
    path: "Path",
    timeline: list,
    *,
    constraints: list | None = None,
    decisions: list | None = None,
) -> None:
    try:
        import rerun as rr
    except ImportError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_dp3_reach_policy_rollout", spawn=False)
    rr.save(str(path))
    
    # Inject GraspGen pitchforks into the main Rerun session!
    if CURRENT_RERUN_DATA is not None:
        _graspgen_rerun_data = CURRENT_RERUN_DATA
        rr.set_time_sequence("step", 0)
        _log_graspgen_rerun(
            _graspgen_rerun_data["all_grasps"],
            _graspgen_rerun_data["all_scores"],
            _graspgen_rerun_data["best_idx"],
            _graspgen_rerun_data["object_cloud"],
            episode_index=_graspgen_rerun_data.get("episode_index", 0),
        )

    if constraints:
        from pg3d.viz.constraints import avoid_region_line_visuals, cartesian_pose_line_visuals

        rr.set_time_sequence("step", 0)
        for visual in avoid_region_line_visuals(constraints):
            rr.log(
                f"world/constraints/{visual.name}",
                rr.LineStrips3D(visual.line_strips, colors=visual.color),
                static=True,
            )
        for constraint in constraints:
            if type(constraint).__name__ == "CartesianPoseConstraint":
                for visual in cartesian_pose_line_visuals(constraint):
                    rr.log(
                        f"world/constraints/{visual.name}",
                        rr.LineStrips3D(visual.line_strips, colors=visual.color),
                        static=True,
                    )

    for step_idx, entry in enumerate(timeline):
        rr.set_time_sequence("step", step_idx)
        import numpy as np
        valid = np.asarray(entry["point_valid_mask"], dtype=bool)
        points = np.asarray(entry["point_cloud"], dtype=np.float32)[valid]
        if points.size:
            rr.log("world/point_cloud", rr.Points3D(points, colors=[180, 180, 180]))
            robot_points = points[np.asarray(entry["robot_mask"], dtype=bool)[valid]]
            if robot_points.size:
                rr.log("world/robot_points", rr.Points3D(robot_points, colors=[0, 128, 255]))
        target = np.asarray(entry["target_position"], dtype=np.float32).reshape(1, 3)
        if np.all(np.isfinite(target)):
            rr.log("world/goal", rr.Points3D(target, colors=[0, 255, 0]))
        tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].reshape(1, 3)
        if np.all(np.isfinite(tcp)):
            rr.log("world/tcp", rr.Points3D(tcp, colors=[255, 220, 0]))

        if decisions is not None:
            active_decision = None
            for d_step, d in decisions:
                if d_step <= step_idx:
                    active_decision = d
                else:
                    break
            if active_decision is not None and getattr(active_decision, "result", None) is not None:
                result = active_decision.result
                rejected_paths = []
                for candidate in result.candidates:
                    if candidate is not result.selected:
                        path = np.asarray(candidate.rollout.eef_path, dtype=np.float32)
                        if path.ndim == 2 and path.shape[0] >= 2 and path.shape[1] == 3:
                            rejected_paths.append(path)
                if rejected_paths:
                    rr.log(
                        "world/predicted_trajectories/rejected",
                        rr.LineStrips3D(rejected_paths, colors=[100, 100, 100, 128], radii=0.001),
                    )
                selected_path = np.asarray(result.selected.rollout.eef_path, dtype=np.float32)
                if selected_path.ndim == 2 and selected_path.shape[0] >= 2 and selected_path.shape[1] == 3:
                    rr.log(
                        "world/predicted_trajectories/selected",
                        rr.LineStrips3D([selected_path], colors=[0, 255, 255, 255], radii=0.003),
                    )
    rr.disconnect()


# Re-register the plain "PG3DReach-RealObstacle-v0" env in case only this
# script is imported (obstacle_envs.py only registers the XArm7 variants).
@register_env("PG3DReach-RealObstacle-v0", max_episode_steps=100)
class _PG3DReachRealObstacleEnv(PG3DReachEnv):
    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        self.obstacle = actors.build_box(
            self.scene,
            half_sizes=[0.03, 0.03, 0.15],
            color=[0.0, 0.0, 1.0, 1.0],
            name="obstacle",
            body_type="kinematic",
        )
        try:
            import sapien.render as sr
            for attr in ("start_site", "goal_site"):
                actor = getattr(self, attr, None)
                if actor is not None:
                    for obj in getattr(actor, "_objs", [actor]):
                        body = obj.find_component_by_type(sr.RenderBodyComponent)
                        if body is not None:
                            if hasattr(body, 'set_visibility'):
                                body.set_visibility(0.0)
                            elif hasattr(body, 'set_visible'):
                                body.set_visible(False)
                            else:
                                body.disable()
        except ImportError:
            pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            start_pos = self.agent.tcp_pose.p
            goal_pos = self.goal_site.pose.p
            mid_pos = (start_pos + goal_pos) / 2.0
            mid_pos[:, 2] = 0.15
            self.obstacle.set_pose(Pose.create_from_pq(mid_pos))


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EvalMethod = Literal["base", "rejection", "reranking"]
GeometryMode = Literal["fast", "exact"]
Entry = dict[str, np.ndarray | bool | float]
_PROJECTION_OVERLAY_Z_RANGE = (0.0, 0.5)

logger = logging.getLogger(__name__)


# ===========================================================================
#  GraspGen utilities  (everything NEW compared to pose_steering eval)
# ===========================================================================

def _make_marker_spheres_strictly_virtual(env: Any) -> None:
    """Strip visual components from start_site and goal_site.

    This makes them strictly virtual: they retain their coordinate frame and
    pose so the policy can still compute tcp_to_goal_pos, but they have absolutely
    no visual geometry. They become fully invisible to the depth
    cameras and will never contaminate the point cloud.
    """
    try:
        import sapien.render as sr
    except ImportError:
        return

    unwrapped = getattr(env, "unwrapped", env)
    stripped = []
    for attr in ("start_site", "goal_site"):
        actor = getattr(unwrapped, attr, None)
        if actor is not None:
            try:
                for obj in getattr(actor, "_objs", [actor]):
                    body = obj.find_component_by_type(sr.RenderBodyComponent)
                    if body is not None:
                        if hasattr(body, 'set_visibility'):
                            body.set_visibility(0.0)
                        elif hasattr(body, 'set_visible'):
                            body.set_visible(False)
                        else:
                            body.disable()
                stripped.append(attr)
            except Exception as exc:
                print(f"[GraspGen] WARNING: could not strip {attr}: {exc}", flush=True)

    if stripped:
        print(f"[GraspGen] Made strictly virtual in sim_env: {stripped}", flush=True)


def load_graspgen_sampler(config_path: str | Path) -> Any:
    """Load GraspGenSampler from a Robotiq YAML config.

    Confirmed API (from server probe):
        load_grasp_cfg(config_path) -> OmegaConf DictConfig
        GraspGenSampler(cfg)        -> sampler with .model on GPU
    """
    if not _GRASPGEN_AVAILABLE:
        raise ImportError(
            "grasp_gen is not installed. "
            "Activate the GraspGen venv before running this script.\n"
            f"  (Import error: {_graspgen_err})"  # type: ignore[name-defined]
        )
    
    # User requested to disable outlier removal since we only have 1 object.
    # We monkeypatch the graspmoe module directly here.
    import grasp_gen.samplers.graspmoe as graspmoe_mod
    if hasattr(graspmoe_mod, "_statistical_outlier_removal"):
        graspmoe_mod._statistical_outlier_removal = lambda pts, **kwargs: pts

    config_path = str(config_path)
    print(f"[GraspGen] Loading config: {config_path}", flush=True)
    cfg = load_grasp_cfg(config_path)
    sampler = GraspGenSampler(cfg)
    gripper = getattr(getattr(cfg, "data", cfg), "gripper_name", "unknown")
    print(f"[GraspGen] Sampler ready. gripper={gripper}", flush=True)
    return sampler


def _get_object_actor_xyz(env: Any, *, grasp_actor_name: str | None = None) -> np.ndarray | None:
    """Return the world-frame centroid of the primary graspable actor, or None.

    Priority order:
      1. ``env.unwrapped.cheezit``  — jstbanana-v0's single object.
      2. Named actor via ``grasp_actor_name``.
      3. ``env.unwrapped.ycb_objects[0]``  — kitchen/multi-object envs.
    Returns None when none of the above are available.
    """
    unwrapped = getattr(env, "unwrapped", env)

    # 1. jstbanana-v0 direct attribute
    cheezit = getattr(unwrapped, "cheezit", None)
    if cheezit is not None:
        try:
            p = cheezit.pose.p
            if hasattr(p, "__getitem__"):
                p = p[0]  # batched: (1, 3) → (3,)
            return np.asarray(p).flatten()[:3].astype(np.float32)
        except Exception as exc:
            print(f"[GraspGen] Warning: cheezit.pose.p failed ({exc})", flush=True)

    # 2. Named actor (fallback)
    if grasp_actor_name is not None:
        try:
            actor = next(
                a for a in getattr(unwrapped, "scene", unwrapped).actors
                if getattr(a, "name", "") == grasp_actor_name
            )
            p = actor.pose.p
            if hasattr(p, "__getitem__"):
                p = p[0]
            return np.asarray(p).flatten()[:3].astype(np.float32)
        except Exception:
            pass

    # 3. ycb_objects[0]
    ycb = getattr(unwrapped, "ycb_objects", None)
    if ycb and len(ycb) > 0:
        try:
            p = ycb[0].pose.p
            if hasattr(p, "__getitem__"):
                p = p[0]
            return np.asarray(p).flatten()[:3].astype(np.float32)
        except Exception:
            pass

    return None


def _get_marker_positions(env: Any) -> list[np.ndarray]:
    """Return world-frame XYZ of all kinematic marker spheres (start_site, goal_site).

    These spheres are rendered by the cameras and appear in the point cloud even
    though they have no physics collision.  They must be removed from the crop
    before passing it to GraspGen, otherwise the spurious points shift the
    cloud centroid and the returned grasp translations land in the wrong place.

    Returns a list of (3,) float32 arrays (one per found marker).
    """
    markers: list[np.ndarray] = []
    unwrapped = getattr(env, "unwrapped", env)
    for attr in ("start_site", "goal_site"):
        actor = getattr(unwrapped, attr, None)
        if actor is None:
            continue
        try:
            p = actor.pose.p
            if hasattr(p, "__getitem__"):
                p = p[0]
            xyz = np.asarray(p).flatten()[:3].astype(np.float32)
            markers.append(xyz)
        except Exception:
            pass
    return markers


def _crop_object_pointcloud(
    scene_cloud: np.ndarray,
    robot_mask: np.ndarray,
    target_xyz: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Crop the scene point cloud to a sphere around ``target_xyz``.

    Args:
        scene_cloud      : (N, 3) world-frame scene PC from ManiSkill cameras.
        robot_mask       : (N,)   True where point belongs to the robot.
        target_xyz       : (3,)   world-frame centroid (from actor pose).
        radius           : sphere crop radius in metres (default 0.10 m).

    Returns:
        (M, 3) float32 object-only point cloud (may be empty).
    """
    env_cloud = scene_cloud[~robot_mask]
    # Remove zero-padded placeholder points.
    norms = np.linalg.norm(env_cloud, axis=1)
    env_cloud = env_cloud[norms > 1e-3]
    if env_cloud.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Crop to the object bounding sphere.
    dists = np.linalg.norm(env_cloud - target_xyz.reshape(1, 3), axis=1)
    env_cloud = env_cloud[dists < float(radius)]

    return env_cloud.astype(np.float32)


def _run_graspgen(
    sampler: Any,
    object_crop: np.ndarray,
    *,
    grasp_threshold: float = 0.8,
    num_grasps: int = 200,
    episode_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GraspGen MoE on the object crop.

    Returns:
        all_grasps : (N, 4, 4) float32 SE(3) matrices in **world frame**.
        all_scores : (N,)      float32 discriminator scores in [0, 1].
    """
    # ── Coordinate pipeline ───────────────────────────────────────────────────
    # 'object_crop' is in ManiSkill WORLD frame.
    # GraspGen models are trained on origin-centered point clouds. If we pass
    # world-frame coordinates (which can be far from the origin), the model
    # will predict garbage or grasps floating at the origin.
    # We MUST center the crop here, and re-add the centroid to the outputs.
    extent = object_crop.max(axis=0) - object_crop.min(axis=0)
    centroid = object_crop.mean(axis=0)
    centered_crop = object_crop - centroid

    print(
        f"\n[GraspGen DEBUG] Episode {episode_index} — coordinate pipeline\n"
        f"  crop shape          : {object_crop.shape}\n"
        f"  crop min  (world)   : {object_crop.min(axis=0).tolist()}\n"
        f"  crop max  (world)   : {object_crop.max(axis=0).tolist()}\n"
        f"  centroid  (world)   : {centroid.tolist()}\n"
        f"  local extent (m)    : {extent.tolist()}\n"
        f"  (passing CENTERED crop; will add centroid back to outputs)",
        flush=True,
    )

    result = run_graspmoe(
        centered_crop,
        sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=num_grasps - 1,
        num_yaws=36,
        z_offsets_cm=(-8, -6, -4, -2, 0),
        outlier_threshold=0.014,
        outlier_k=20,
        obb_mode="advanced",
        skip_obb_rule="auto",
        obb_density="sparse",
        obb_position_spacing_m=0.01,
    )
    grasps_diff = result["grasps_diff"]   # (Nd, 4, 4) — translations in LOCAL frame
    scores_diff = result["scores_diff"]
    grasps_obb  = result["grasps_obb"]
    scores_obb  = result["scores_obb"]

    # Debug: show a sample local grasp translation before centroid is added
    if grasps_diff.shape[0] > 0:
        sample = grasps_diff[0, :3, 3]
        print(
            f"  sample grasp[0] local (pre-centroid): {sample.tolist()}",
            flush=True,
        )

    # Re-add centroid to convert translations from local → world frame.
    if grasps_diff.shape[0] > 0:
        grasps_diff[:, :3, 3] += centroid
    if grasps_obb.shape[0] > 0:
        grasps_obb[:, :3, 3] += centroid

    if grasps_diff.shape[0] > 0 and grasps_obb.shape[0] > 0:
        all_grasps = np.concatenate([grasps_diff, grasps_obb], axis=0)
        all_scores = np.concatenate([scores_diff, scores_obb], axis=0)
    elif grasps_diff.shape[0] > 0:
        all_grasps, all_scores = grasps_diff, scores_diff
    elif grasps_obb.shape[0] > 0:
        all_grasps, all_scores = grasps_obb, scores_obb
    else:
        all_grasps = np.zeros((0, 4, 4), dtype=np.float32)
        all_scores = np.zeros((0,),      dtype=np.float32)

    print(
        f"[GraspGen] Episode {episode_index}: "
        f"diffusion={grasps_diff.shape[0]}  OBB={grasps_obb.shape[0]}  "
        f"total={all_grasps.shape[0]}  "
        f"skipped_obb={result['skipped_obb']}",
        flush=True,
    )
    return all_grasps, all_scores


def _apply_z_offset(grasp_T: np.ndarray, z_offset: float) -> np.ndarray:
    """Shift the grasp contact point along the approach axis.

    The approach axis is the gripper's local Z column (T[:3, 2]).
    A positive z_offset moves the contact point BACK along the approach axis
    (further from the surface), which corrects for a gripper that reaches
    further than Robotiq 2F-140.

    Args:
        grasp_T  : (4, 4) SE(3) grasp matrix in world frame.
        z_offset : metres; 0.0 for Robotiq 2F-140 (no change needed).

    Returns:
        (4, 4) SE(3) matrix with adjusted translation.
    """
    if abs(z_offset) < 1e-6:
        return grasp_T
    T = grasp_T.copy().astype(np.float64)
    approach_axis = T[:3, 2]   # local Z column = approach direction
    T[:3, 3] -= z_offset * approach_axis
    return T.astype(np.float32)


def _rot3x3_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a wxyz quaternion (float32).

    Uses scipy.spatial.transform.Rotation (already a dependency).
    """
    from scipy.spatial.transform import Rotation as R_scipy
    quat_xyzw = R_scipy.from_matrix(rot.astype(np.float64)).as_quat()
    # scipy returns xyzw; CartesianPoseConstraint expects wxyz
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Viser / Rerun pitchfork grasp visualisation helpers
# ---------------------------------------------------------------------------

# Gripper geometry constants used for the pitchfork visualisation.
# These match the Robotiq 2F-140 finger geometry that GraspGen was trained on.
_GRASP_VIS_APPROACH_LEN  = 0.06   # length of the blue approach arrow (m)
_GRASP_VIS_FINGER_WIDTH  = 0.14   # Robotiq 2F-140 full span (140 mm)
_GRASP_VIS_FINGER_DEPTH  = 0.065  # Robotiq 2F-140 finger length approx (65 mm)


def _pitchfork_lines_for_grasp(
    grasp_T: np.ndarray,
    *,
    approach_len: float = _GRASP_VIS_APPROACH_LEN,
    finger_width: float = _GRASP_VIS_FINGER_WIDTH,
    finger_depth: float = _GRASP_VIS_FINGER_DEPTH,
) -> list[np.ndarray]:
    """Return line segments that form a T-bar / pitchfork for one grasp SE(3)."""
    grasp_T = np.asarray(grasp_T, dtype=np.float32)
    origin   = grasp_T[:3, 3]           # TCP / grasp contact centre
    approach = grasp_T[:3, 2]           # local Z column (points FROM base TO TCP)
    finger   = grasp_T[:3, 0]           # local X column

    # TCP is exactly at 'origin'.
    # The base of the fingers (crossbar) is 'finger_depth' behind the TCP.
    finger_root = origin - approach * finger_depth
    
    # Handle / approach line (from the finger root, backwards)
    handle_end = finger_root - approach * approach_len
    
    # Crossbar
    half_width = finger_width / 2.0
    left_root = finger_root - finger * half_width
    right_root = finger_root + finger * half_width
    
    # Prongs (from the root, forwards to the tips at the TCP plane)
    left_tip = origin - finger * half_width
    right_tip = origin + finger * half_width
    
    lines = [
        np.stack([finger_root, handle_end], axis=0),   # handle
        np.stack([left_root, right_root], axis=0),     # crossbar
        np.stack([left_root, left_tip], axis=0),       # left prong
        np.stack([right_root, right_tip], axis=0),     # right prong
    ]
    return [l.astype(np.float32) for l in lines]


def _log_graspgen_rerun(
    all_grasps: np.ndarray,
    all_scores: np.ndarray,
    best_idx: int,
    object_cloud: np.ndarray,
    *,
    episode_index: int,
) -> None:
    """Log all grasp candidates + object cloud to an open Rerun session.

    Called inside _build_graspgen_constraint BEFORE save_rerun_timeline;
    uses static=True so the visuals persist across all timesteps.
    """
    try:
        import rerun as rr
    except ImportError:
        return

    # Object point cloud (centred)
    if object_cloud.shape[0] > 0:
        rr.log(
            "world/graspgen/object_cloud",
            rr.Points3D(object_cloud, colors=[255, 200, 50]),
            static=True,
        )

    # All candidates — green lines
    # (Disabled: user requested ONLY the best grasp to be shown in Rerun)

    # Best grasp — bright yellow lines, thicker
    best_lines = _pitchfork_lines_for_grasp(all_grasps[best_idx])
    rr.log(
        "world/graspgen/best_grasp",
        rr.LineStrips3D(best_lines, colors=[255, 255, 0, 255], radii=0.003),
        static=True,
    )
    # Best grasp origin point
    rr.log(
        "world/graspgen/best_grasp_origin",
        rr.Points3D(
            all_grasps[best_idx:best_idx + 1, :3, 3],
            colors=[255, 80, 0], radii=0.008,
        ),
        static=True,
    )
    print(
        f"[GraspGen] Logged {all_grasps.shape[0]} candidates to Rerun "
        f"(episode {episode_index}, best_idx={best_idx})",
        flush=True,
    )


def _show_graspgen_viser(
    all_grasps: np.ndarray,
    all_scores: np.ndarray,
    best_idx: int,
    object_cloud: np.ndarray,
    *,
    episode_index: int,
    block: bool = True,
) -> None:
    """Open a local Viser window with the object PC and all grasp candidates.

    Visualises each grasp as a pitchfork (approach + finger lines) coloured
    from blue (low score) to green (high score). The best grasp is bright red.
    Set block=True to pause until the user closes the window (default), or
    block=False to return immediately (window keeps running in the background).
    """
    try:
        import viser
        import viser.transforms as vtf
    except ImportError:
        print(
            "[GraspGen] viser not installed — skipping debug window. "
            "Install with: pip install viser",
            flush=True,
        )
        return

    server = viser.ViserServer()
    server.scene.world_axes.visible = False

    # Object point cloud
    if object_cloud.shape[0] > 0:
        server.scene.add_point_cloud(
            "object_cloud",
            points=object_cloud,
            colors=np.tile([255, 200, 50], (object_cloud.shape[0], 1)),
            point_size=0.004,
        )

    # Score range for colour mapping
    s_min = float(all_scores.min())
    s_max = float(all_scores.max())
    s_range = max(s_max - s_min, 1e-6)

    for i, (grasp_T, score) in enumerate(zip(all_grasps, all_scores)):
        pos = grasp_T[:3, 3]

        if i == best_idx:
            color = (255, 80, 0)
            node_name = "grasps/best"
            line_width = 3.0
            radius = 0.015
        else:
            normalized_score = float((score - s_min) / s_range)
            # Interpolate between blue (lowest score) and green (highest score)
            color = (0, int(255 * normalized_score), int(255 * (1 - normalized_score)))
            node_name = f"grasps/candidate_{i}"
            line_width = 1.0
            radius = 0.005

        # ── 1. Plain sphere at TCP position ──────────────────────────────────
        # This makes it immediately obvious whether the POSITION is correct
        # regardless of orientation/pitchfork rendering artifacts.
        server.scene.add_icosphere(
            f"{node_name}/tcp_sphere",
            radius=radius,
            color=color,
            position=pos,
        )

        # ── 2. Pitchfork lines (approach + finger bars) ───────────────────────
        lines = _pitchfork_lines_for_grasp(grasp_T)
        for j, seg in enumerate(lines):
            server.scene.add_line_segments(
                f"{node_name}/{j}",
                points=np.expand_dims(seg, 0),
                colors=color,
                line_width=line_width,
            )

    # ── 3. Actor centroid marker (magenta) ────────────────────────────────────
    # Shows where _get_object_actor_xyz reported the object to be.
    # If this overlaps the cheezit point cloud, the centroid read is correct.
    if object_cloud.shape[0] > 0:
        cloud_centroid = object_cloud.mean(axis=0)
        server.scene.add_icosphere(
            "debug/cloud_centroid",
            radius=0.012,
            color=(220, 0, 220),
            position=cloud_centroid,
        )

    print(
        f"[GraspGen Viser] Episode {episode_index}: showing {all_grasps.shape[0]} grasps. "
        f"Open browser at http://localhost:8080 — close the window or Ctrl+C to continue.",
        flush=True,
    )
    if block:
        try:
            while True:
                import time
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    server.stop()


def _build_graspgen_constraint(
    env: Any,
    *,
    spec: RolloutSpec,
    crop_config: PointCloudCropConfig,
    graspgen_sampler: Any,
    args: argparse.Namespace,
    zarr_context: dict[str, Any] | None = None,
) -> tuple[list[Any], None]:
    """Reset the env and build a CartesianPoseConstraint from GraspGen.

    Key changes vs the old version
    --------------------------------
    * Object pose: read directly from ``env.unwrapped.cheezit`` (jstbanana-v0)
      rather than from ``entry['target_position']`` (which is the goal_site).
    * Visualisation: if ``--graspgen-viser`` is set, opens a blocking Viser
      window with all grasp candidates; if ``--rerun`` is set, logs pitchforks
      to the active Rerun session.

    Returns (constraints, None) — the None matches the PendingObstacleSpawn
    slot from the original signature so the caller loop doesn't need changes.
    """
    # --- 1. Reset ---
    # For jstbanana-v0 we explicitly want to test on freshly randomized
    # objects (ignoring the dataset state), otherwise the cheezit box
    # is left at the origin because it doesn't exist in the legacy dataset.
    is_banana = getattr(env.unwrapped, "_JSTBANANA_Z", None) is not None
    if zarr_context is not None and not is_banana:
        obs, info = _reset_to_zarr_episode(env, rollout_seed=spec.seed, zarr_context=zarr_context)
    else:
        obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})

    # --- 2. Zero-action step: forces renderer to re-render after _initialize_episode ---
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    action_mode_str = str(getattr(args, "action_mode", "abs_joint"))
    if action_mode_str == "abs_joint":
        current_qpos = np.asarray(env.unwrapped.agent.robot.get_qpos())
        qpos_flat = current_qpos.reshape(-1)
        za_flat = zero_action.reshape(-1)
        dof = min(len(qpos_flat), len(za_flat))
        za_flat[:dof] = qpos_flat[:dof]
        zero_action = za_flat.reshape(zero_action.shape)
    obs, _, _, _, info = env.step(zero_action)

    # --- 3. Get scene point cloud ---
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    if zarr_context is not None:
        entry = _apply_zarr_initial_entry(entry, zarr_context)

    scene_cloud = np.asarray(entry["point_cloud"], dtype=np.float32).reshape(-1, 3)
    robot_mask  = np.asarray(entry["robot_mask"],  dtype=bool).reshape(-1)

    # --- 3a. Resolve object position: direct actor pose takes priority ---
    # For jstbanana-v0 the cheezit object is placed randomly in the workspace;
    # entry['target_position'] (the goal_site) follows the object because the
    # env moves goal_site to the cheezit centroid, BUT reading directly from
    # the actor is cleaner and avoids any timing races.
    grasp_actor_name = getattr(args, "grasp_actor_name", None)
    actor_xyz = _get_object_actor_xyz(env, grasp_actor_name=grasp_actor_name)

    if actor_xyz is not None:
        target_xyz = actor_xyz
        print(
            f"[GraspGen] Episode {spec.output_index}: "
            f"object actor pose → {target_xyz.tolist()}",
            flush=True,
        )
    else:
        # Fallback: use goal_site / target_position (pose steering behaviour)
        target_xyz = np.asarray(entry["target_position"], dtype=np.float32).reshape(3)
        print(
            f"[GraspGen] Episode {spec.output_index}: "
            f"no actor found, using goal_site target_position → {target_xyz.tolist()}",
            flush=True,
        )

    # --- 4. Crop to object ---
    # The start_site and goal_site markers are strictly virtual and invisible,
    # so we DO NOT need to extract their positions and filter them out anymore.
    # Filtering around goal_site was accidentally deleting the actual object points!
    crop_radius = float(getattr(args, "grasp_object_crop_radius", 0.10))
    object_crop = _crop_object_pointcloud(
        scene_cloud, robot_mask, target_xyz, crop_radius
    )

    print(
        f"[GraspGen] Episode {spec.output_index}: "
        f"object_crop_points={object_crop.shape[0]}  "
        f"(radius={crop_radius:.2f}m around {target_xyz.tolist()})",
        flush=True,
    )

    # --- 5. Run GraspGen ---
    grasp_threshold = float(getattr(args, "graspgen_threshold", 0.8))
    num_grasps      = int(getattr(args, "graspgen_num_grasps", 200))

    if object_crop.shape[0] < 10:
        print(
            f"[GraspGen] Episode {spec.output_index}: WARNING — only "
            f"{object_crop.shape[0]} object points (< 10); "
            "falling back to goal-position-only constraint (no orientation).",
            flush=True,
        )
        constraint = CartesianPoseConstraint(
            target_position=target_xyz,
            target_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            position_tolerance=float(getattr(args, "grasp_position_tolerance", 0.03)),
            rotation_tolerance=3.15,
            weight=float(getattr(args, "grasp_weight", 2.0)),
            name="graspgen_fallback_position_only",
        )
        return [constraint], None

    all_grasps, all_scores = _run_graspgen(
        graspgen_sampler,
        object_crop,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        episode_index=spec.output_index,
    )

    if all_grasps.shape[0] == 0:
        print(
            f"[GraspGen] Episode {spec.output_index}: WARNING — no grasps returned "
            "(below threshold or empty cloud). Falling back to position-only constraint.",
            flush=True,
        )
        constraint = CartesianPoseConstraint(
            target_position=target_xyz,
            target_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            position_tolerance=float(getattr(args, "grasp_position_tolerance", 0.03)),
            rotation_tolerance=3.15,
            weight=float(getattr(args, "grasp_weight", 2.0)),
            name="graspgen_fallback_position_only",
        )
        return [constraint], None

    # --- 6. Pick best grasp (Top-Down heuristic) ---
    # We want grasps that approach from the top. The approach vector is the local Z
    # axis of the grasp (column 2 of the rotation matrix), which points from the
    # gripper base to the TCP. A perfectly top-down grasp points along world -Z
    # (so approach_z = -1). We penalise grasps that approach from the side/bottom.
    approach_zs = all_grasps[:, 2, 2]
    
    # Combined score = GraspGen score (0 to 1) - 0.2 * approach_z
    # This gives up to a +0.2 boost to perfectly top-down grasps, and penalises
    # bottom-up grasps by -0.2.
    combined_scores = all_scores - 0.2 * approach_zs
    
    best_idx   = int(np.argmax(combined_scores))
    best_score = float(all_scores[best_idx])
    best_grasp = all_grasps[best_idx].astype(np.float32)  # (4, 4)

    print(
        f"[GraspGen] Episode {spec.output_index}: "
        f"total_grasps={all_grasps.shape[0]}  "
        f"best_idx={best_idx}  "
        f"best_score={best_score:.4f} (combined={float(combined_scores[best_idx]):.4f}, approach_z={float(approach_zs[best_idx]):.2f})  "
        f"raw_score_range=[{float(all_scores.min()):.4f}, {float(all_scores.max()):.4f}]",
        flush=True,
    )

    # --- 7. Diagnostic: raw Robotiq-frame grasp ---
    raw_pos  = best_grasp[:3, 3]
    raw_rot  = best_grasp[:3, :3]
    raw_quat = _rot3x3_to_quat_wxyz(raw_rot)
    approach = raw_rot[:, 2]  # local Z = approach axis

    # Compute distance from grasp TCP to the object actor centroid.
    # This should be in the range [0, ~0.15 m] for a well-placed grasp.
    # If it's >> 0.2 m the centroid estimate is wrong (marker contamination).
    dist_to_actor = float(np.linalg.norm(raw_pos - target_xyz))

    print(
        f"[GraspGen] Episode {spec.output_index}: raw grasp (world frame):\n"
        f"  object actor xyz (world) = {target_xyz.tolist()}\n"
        f"  grasp  TCP   xyz (world) = {raw_pos.tolist()}\n"
        f"  distance actor→grasp     = {dist_to_actor:.4f} m  ← should be < 0.20 m\n"
        f"  quaternion    = {raw_quat.tolist()}  (wxyz)\n"
        f"  approach_axis = {approach.tolist()}  (local Z col of rotation)",
        flush=True,
    )

    # --- 7b. Debug visualisations (Viser + Rerun pitchforks) ---
    if getattr(args, "graspgen_viser", False):
        _show_graspgen_viser(
            all_grasps, all_scores, best_idx, object_crop,
            episode_index=spec.output_index,
            block=True,
        )

    # Rerun pitchfork logging: deferred — logged inside save_rerun_timeline.
    # Store on args so run_eval_episode can pass them through.
    if getattr(args, "rerun", False):
        # Attach the grasp data as a per-episode attribute on args so that the
        # Rerun writer in run_eval_episode can call _log_graspgen_rerun.
        # We use a dict keyed by output_index to handle multiple episodes.
        if not hasattr(args, "_graspgen_rerun_data"):
            args._graspgen_rerun_data = {}
        args._graspgen_rerun_data[spec.output_index] = {
            "all_grasps": all_grasps,
            "all_scores": all_scores,
            "best_idx": best_idx,
            "object_cloud": object_crop,
        }

    # --- 8. Apply Z-offset for XArm7 gripper geometry ---
    z_offset = float(getattr(args, "graspgen_z_offset", 0.0))
    adjusted = _apply_z_offset(best_grasp, z_offset)
    adj_pos  = adjusted[:3, 3]
    adj_rot  = adjusted[:3, :3]
    adj_quat = _rot3x3_to_quat_wxyz(adj_rot)

    if getattr(args, "rerun", False):
        args._graspgen_rerun_data[spec.output_index]["final_grasp_pos"] = adj_pos
        args._graspgen_rerun_data[spec.output_index]["final_grasp_quat"] = adj_quat

    # --- 8b. Create Pre-Grasp Pose ---
    approach_offset = float(getattr(args, "grasp_approach_offset", 0.10))
    # approach vector is the local Z axis of the adjusted rotation
    approach_vec = adj_rot[:, 2]
    # offset backward (negative approach_vec)
    pregrasp_pos = adj_pos - approach_offset * approach_vec
    
    if abs(approach_offset) > 1e-6:
        print(
            f"[GraspGen] Episode {spec.output_index}: applying approach offset={approach_offset:.4f}m:\n"
            f"  pre-grasp position = {pregrasp_pos.tolist()}\n",
            flush=True,
        )

    # --- 9. Build CartesianPoseConstraint ---
    pos_tol  = float(getattr(args, "grasp_position_tolerance", 0.02))
    rot_tol  = float(getattr(args, "grasp_rotation_tolerance", 0.35))
    weight   = float(getattr(args, "grasp_weight", 2.0))

    constraint = CartesianPoseConstraint(
        target_position=pregrasp_pos,
        target_orientation=adj_quat,   # (4,) wxyz quaternion
        position_tolerance=pos_tol,
        rotation_tolerance=rot_tol,
        weight=weight,
        name="graspgen_reach",
        metadata={
            "episode_output_index": spec.output_index,
            "episode_seed": spec.seed,
            "graspgen_best_score": best_score,
            "graspgen_total_grasps": int(all_grasps.shape[0]),
            "graspgen_best_idx": best_idx,
            "z_offset": z_offset,
        },
    )
    print(
        f"[GraspGen] Episode {spec.output_index}: "
        f"CartesianPoseConstraint  pos_tol={pos_tol:.3f}m  "
        f"rot_tol={rot_tol:.3f}rad  weight={weight:.1f}",
        flush=True,
    )
    constraints: list[Any] = [constraint]

    # Optionally stack a joint posture constraint (same as pose steering eval)
    if getattr(args, "posture_target_joints", None) is not None:
        posture_constraint = JointPostureConstraint(
            target_q=np.array(args.posture_target_joints, dtype=np.float32),
            weight=float(args.posture_weight),
            eval_timestep=args.posture_eval_timestep,
        )
        constraints.append(posture_constraint)

    return constraints, None   # None = no PendingObstacleSpawn


# ===========================================================================
#  Shared infrastructure — identical to eval_pointcloud_pose_steering_reach
#  (DP3ChunkPolicyAdapter, _repeat_obs_window_to_torch, etc.)
# ===========================================================================

class DummyRegion:
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.full(points.shape[:-1], 1000.0, dtype=np.float32)


@dataclass
class PointCloudCollisionConstraint:
    """Kept for compatibility if anyone passes --constraint-type pointcloud_collision."""
    obstacle_points: np.ndarray
    target: str = "robot"
    margin: float = 0.05
    weight: float = 100.0
    constraint_type: str = "pointcloud_collision"
    name: str = "pointcloud_collision"

    def __post_init__(self):
        self.region = DummyRegion()

    def cost(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> dict[str, float]:
        if self.obstacle_points.size == 0:
            return {self.name: 0.0}
        if self.target == "eef":
            points = rollout.eef_path
            clouds = [c for c in rollout.robot_point_clouds if c.size]
            if clouds:
                points = np.concatenate([points] + clouds, axis=0)
        elif self.target == "robot":
            clouds = [c for c in rollout.robot_point_clouds if c.size]
            if not clouds:
                return {self.name: 0.0}
            points = np.concatenate(clouds, axis=0)
        else:
            return {self.name: 0.0}
        dists = cdist(points, self.obstacle_points)
        min_dists = np.min(dists, axis=1)
        violation = np.maximum(self.margin - min_dists, 0.0)
        max_v = float(np.max(violation)) if violation.size else 0.0
        mean_v = float(np.mean(violation)) if violation.size else 0.0
        return {
            self.name: float(self.weight) * (max_v + 0.5 * mean_v),
            f"{self.name}/max_violation": max_v,
            f"{self.name}/mean_violation": mean_v,
        }

    def satisfied(self, rollout: ImaginedRollout, scene: SceneContext | None = None) -> bool:
        return self.cost(rollout, scene).get(f"{self.name}/max_violation", 0.0) <= 1e-6

    def to_json(self) -> dict[str, Any]:
        return {"type": self.constraint_type,
                "obstacle_points_count": int(self.obstacle_points.shape[0])}


def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= n_samples:
        return pts
    selected: list[int] = [0]
    min_dists = np.full(pts.shape[0], np.inf, dtype=np.float32)
    for _ in range(n_samples - 1):
        last = pts[selected[-1]]
        dists = np.linalg.norm(pts - last, axis=1).astype(np.float32)
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))
    return pts[np.array(selected, dtype=np.int64)]


# ---------------------------------------------------------------------------
# Torch helpers (identical to pose steering script)
# ---------------------------------------------------------------------------

def _repeat_obs_window_to_torch(
    window: list[Entry],
    *,
    k: int,
    device: torch.device,
    goal_marker_points: int = 0,
    goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> dict[str, torch.Tensor]:
    batch = obs_window_to_torch(
        window,
        device=device,
        goal_marker_points=goal_marker_points,
        goal_marker_radius=goal_marker_radius,
    )
    return {
        key: value.repeat((k, *([1] * (value.ndim - 1))))
        for key, value in batch.items()
    }


def _obs_windows_to_torch(
    windows: list[list[Entry]],
    *,
    device: torch.device,
    goal_marker_points: int = 0,
    goal_marker_radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> dict[str, torch.Tensor]:
    if not windows:
        raise ValueError("windows must not be empty")
    point_cloud = np.stack(
        [np.stack([entry["point_cloud"] for entry in window], axis=0) for window in windows],
        axis=0,
    )
    if goal_marker_points:
        target_position = np.stack(
            [np.stack([entry["target_position"] for entry in window], axis=0) for window in windows],
            axis=0,
        )
        point_cloud = insert_goal_marker_points(
            point_cloud,
            target_position,
            num_points=goal_marker_points,
            radius=goal_marker_radius,
        )
    agent_pos = np.stack(
        [np.stack([entry["agent_pos"] for entry in window], axis=0) for window in windows],
        axis=0,
    )
    goal_xyz = np.stack(
        [np.stack([entry["target_position"] for entry in window], axis=0) for window in windows],
        axis=0,
    )
    ee_position = np.stack(
        [
            np.stack(
                [np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3] for entry in window],
                axis=0,
            )
            for window in windows
        ],
        axis=0,
    )
    return {
        "point_cloud":  torch.from_numpy(point_cloud.astype(np.float32)).to(device),
        "agent_pos":    torch.from_numpy(agent_pos.astype(np.float32)).to(device),
        "goal_xyz":     torch.from_numpy(goal_xyz.astype(np.float32)).to(device),
        "ee_position":  torch.from_numpy(ee_position.astype(np.float32)).to(device),
    }


# ---------------------------------------------------------------------------
# DP3ChunkPolicyAdapter  (identical to pose steering script)
# ---------------------------------------------------------------------------

@dataclass
class EvalDecisionSummary:
    selected_chunk: ActionChunk
    result: ControllerResult | None
    candidate_feasible: int
    candidate_total: int
    selection_reason: str | None


@dataclass
class PendingObstacleSpawn:
    """Kept for API compatibility — not used in graspgen_pick."""
    reference_path: np.ndarray
    trigger_fraction: float
    constraint: AvoidRegion


class DP3ChunkPolicyAdapter:
    """Adapt SimpleDP3.predict_action to the candidate-sampling protocol."""

    def __init__(
        self,
        policy: SimpleDP3,
        *,
        action_mode: ActionMode,
        device: torch.device,
        policy_batch_size: int = 64,
        timer: TimingRecorder | None = None,
        dt: float = 1.0,
    ) -> None:
        self.policy = policy
        self.action_mode = action_mode
        self.device = device
        self.policy_batch_size = int(policy_batch_size)
        self.timer = timer or TimingRecorder(enabled=False)
        self.dt = float(dt)

    def sample_action_chunks(
        self,
        policy_input: list[Entry],
        *,
        k: int,
        rng: np.random.Generator | None = None,
    ) -> list[ActionChunk]:
        if k <= 0:
            raise ValueError("k must be positive")
        with self.timer.time("policy_sampling", windows=1, samples=k):
            batch = _repeat_obs_window_to_torch(
                policy_input,
                k=k,
                device=self.device,
                goal_marker_points=int(getattr(self.policy, "goal_marker_points", 0)),
                goal_marker_radius=float(
                    getattr(self.policy, "goal_marker_radius", DEFAULT_GOAL_MARKER_RADIUS)
                ),
            )
            actions = self._predict_actions(batch)
        return [
            ActionChunk(
                actions=actions[idx].astype(np.float32, copy=True),
                action_mode=self.action_mode,
                dt=self.dt,
                metadata={"candidate_index": idx},
            )
            for idx in range(actions.shape[0])
        ]

    def sample_action_chunks_for_windows(
        self,
        policy_inputs: list[list[Entry]],
        *,
        rng: np.random.Generator | None = None,
    ) -> list[ActionChunk]:
        if not policy_inputs:
            return []
        del rng
        actions: list[np.ndarray] = []
        with self.timer.time("policy_sampling", windows=len(policy_inputs), samples=1):
            for start in range(0, len(policy_inputs), self.policy_batch_size):
                batch_windows = policy_inputs[start : start + self.policy_batch_size]
                batch = _obs_windows_to_torch(
                    batch_windows,
                    device=self.device,
                    goal_marker_points=int(getattr(self.policy, "goal_marker_points", 0)),
                    goal_marker_radius=float(
                        getattr(self.policy, "goal_marker_radius", DEFAULT_GOAL_MARKER_RADIUS)
                    ),
                )
                actions.append(self._predict_actions(batch))
        stacked = np.concatenate(actions, axis=0)
        return [
            ActionChunk(
                actions=stacked[idx].astype(np.float32, copy=True),
                action_mode=self.action_mode,
                dt=self.dt,
                metadata={"candidate_index": idx},
            )
            for idx in range(stacked.shape[0])
        ]

    def _predict_actions(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.inference_mode():
            output = self.policy.predict_action(batch)
            return output["action"].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Misc helpers copied verbatim from pose steering eval
# ---------------------------------------------------------------------------

def _append_path(path: EpisodePath, entry: Entry) -> None:
    path.append_pose(
        tcp_pose=entry["tcp_pose"],
        q=np.asarray(entry["agent_pos"], dtype=np.float32),
        target_distance=float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0]),
    )


def _env_kwargs(
    metadata: dict[str, Any],
    *,
    render_mode: str | None,
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["num_envs"] = 1
    if render_mode is None:
        env_kwargs.pop("render_mode", None)
    else:
        env_kwargs["render_mode"] = render_mode
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    return env_kwargs


def _video_env_factory(
    gym: Any,
    *,
    metadata: dict[str, Any],
    enabled: bool,
    max_episode_steps: int | None = None,
) -> Callable[[], Any] | None:
    if not enabled:
        return None
    env_kwargs = _env_kwargs(metadata, render_mode="rgb_array",
                             max_episode_steps=max_episode_steps)
    def factory() -> Any:
        return gym.make(str(metadata["env_id"]), **env_kwargs)
    return factory


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


def _env_task_name(env: Any) -> str:
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    return str(getattr(spec, "id", "unknown"))


def _env_task_name_from_id(env_id: str) -> str:
    return env_id


def _format_optional(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cuda_sync_fn(device: torch.device) -> Any | None:
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.synchronize


class _null_timer:
    def __enter__(self) -> None:
        return None
    def __exit__(self, *_args: Any) -> bool:
        return False


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _copy_entry(entry: Entry) -> Entry:
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in entry.items()}


def _copy_window(window: list[Entry]) -> list[Entry]:
    return [_copy_entry(entry) for entry in window]


def _constraint_source_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "graspgen_pick",
        "graspgen_config": str(getattr(args, "graspgen_config", None)),
        "graspgen_threshold": float(getattr(args, "graspgen_threshold", 0.8)),
        "graspgen_num_grasps": int(getattr(args, "graspgen_num_grasps", 200)),
        "graspgen_z_offset": float(getattr(args, "graspgen_z_offset", 0.0)),
        "grasp_object_crop_radius": float(getattr(args, "grasp_object_crop_radius", 0.10)),
        "grasp_object_index": int(getattr(args, "grasp_object_index", -1)),
        "grasp_weight": float(getattr(args, "grasp_weight", 2.0)),
        "grasp_position_tolerance": float(getattr(args, "grasp_position_tolerance", 0.02)),
        "grasp_rotation_tolerance": float(getattr(args, "grasp_rotation_tolerance", 0.35)),
    }


def _episode_indices_from_args(
    args: argparse.Namespace,
    *,
    dataset_episode_seeds: list[int],
) -> list[int] | None:
    if getattr(args, "unique_dataset_seeds", False):
        return _unique_seed_episode_indices(
            dataset_episode_seeds,
            max_count=int(args.episodes),
        )
    if getattr(args, "episode_indices_file", None) is not None:
        return _read_episode_indices_file(args.episode_indices_file)
    return getattr(args, "episode_indices", None)


def _unique_seed_episode_indices(
    dataset_episode_seeds: list[int],
    *,
    max_count: int,
) -> list[int]:
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    seen: set[int] = set()
    indices: list[int] = []
    for dataset_idx, seed in enumerate(dataset_episode_seeds):
        if seed in seen:
            continue
        seen.add(seed)
        indices.append(dataset_idx)
        if len(indices) >= max_count:
            break
    if not indices:
        raise ValueError("dataset metadata did not contain any episode seeds")
    return indices


def _read_episode_indices_file(path: Path) -> list[int]:
    indices: list[int] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = int(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} is not an integer episode index") from exc
        if value < 0:
            raise ValueError(f"{path}:{line_no} episode index must be non-negative")
        indices.append(value)
    if not indices:
        raise ValueError(f"{path} did not contain any episode indices")
    return indices


def _zarr_episode_context_with_paths(zarr_root: Any, episode_index: int) -> dict[str, Any]:
    context = _zarr_episode_context(zarr_root, episode_index)
    episode_ends = np.asarray(zarr_root["meta"]["episode_ends"][:], dtype=np.int64)
    episode_start = int(context["episode_start"])
    episode_end = int(episode_ends[episode_index])
    data = zarr_root["data"]
    context["episode_end"] = episode_end
    context["tcp_pose_path"] = np.asarray(
        data["tcp_pose"][episode_start:episode_end],
        dtype=np.float32,
    ).copy()
    return context


def _write_new_timing_events(
    timer: TimingRecorder,
    path: Path,
    *,
    start_index: int,
) -> int:
    if not timer.enabled:
        return start_index
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for idx, event in enumerate(timer.events[start_index:], start=start_index):
            file.write(json.dumps({"index": idx, **event.to_json()}, sort_keys=True) + "\n")
    return len(timer.events)


def _print_timing_summary(timer: TimingRecorder) -> None:
    summary = timer.summary()
    if not summary:
        return
    top = sorted(summary.items(), key=lambda item: item[1]["total"], reverse=True)[:6]
    text = ", ".join(
        f"{name}={values['total']:.2f}s/{int(values['count'])}x" for name, values in top
    )
    print(f"timing: {text}")


def _init_wandb(
    args: argparse.Namespace,
    *,
    metadata: dict[str, Any],
    checkpoint_path: Path,
) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "dataset": str(args.dataset),
                "checkpoint": str(checkpoint_path),
                "env_id": metadata.get("env_id"),
                "methods": list(args.methods),
                "constraint_source": _constraint_source_summary(args),
                "command": "scripts/eval_graspgen_pick.py",
            },
        )
    except Exception as exc:
        if args.wandb_required:
            raise
        print(f"warning: W&B init failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _log_wandb_episode(
    run: Any | None,
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    global_step: int,
) -> None:
    if run is None:
        return
    try:
        metrics = {
            f"episode/{row['method']}/reach_success":
                float(row["reach_success"]),
            f"episode/{row['method']}/constraint_satisfied":
                float(row["constraint_satisfied"]),
            f"episode/{row['method']}/combined_success":
                float(row["combined_success"]),
            f"episode/{row['method']}/final_target_distance":
                row["final_target_distance"],
            "episode/index": row["episode"],
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        with _null_timer():
            run.log(metrics, step=global_step)
    except Exception as exc:
        if args.wandb_required:
            raise
        print(f"warning: W&B episode log failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _maybe_emit_progress(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    timer: TimingRecorder,
    episode_index: int,
    plots: bool,
    run: Any | None,
    args: argparse.Namespace,
    final: bool = False,
) -> None:
    if not rows:
        return
    summarize_metrics(rows)
    if run is None:
        return


def resolve_checkpoint_path(checkpoint: Path | None, checkpoint_dir: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    if checkpoint_dir is not None:
        return latest_reach_checkpoint(checkpoint_dir)
    raise ValueError("either --checkpoint or --checkpoint-dir must be specified")


# ===========================================================================
#  main()  —  same orchestration as pose steering, but calls
#              _build_graspgen_constraint instead of _constraints_for_episode,
#              and loads GraspGenSampler first.
# ===========================================================================

def _execute_open_loop_grasp(
    sim_env: Any,
    video_env: Any,
    frames: list[np.ndarray],
    timeline: list[dict[str, Any]],
    method: str,
):
    import sapien.core as sapien
    import rerun as rr
    from pg3d.utils.arrays import frame_to_numpy as _frame_to_numpy
    from scripts.eval_pointcloud_pose_steering_reach import _render_video_frame
    
    # Get pre-saved final grasp pose from args
    global CURRENT_RERUN_DATA
    if CURRENT_RERUN_DATA is None or "final_grasp_pos" not in CURRENT_RERUN_DATA:
        return
        
    final_grasp_pos = CURRENT_RERUN_DATA["final_grasp_pos"]
    final_grasp_quat = CURRENT_RERUN_DATA["final_grasp_quat"]
    
    print("\n--- [Phase 1] Executing Open-Loop Grasp Approach ---", flush=True)
    
    model = sim_env.agent.robot.create_pinocchio_model()
    link_idx = sim_env.agent.robot.links_map[sim_env.agent.ee_link_name].get_index()
    
    current_qpos = sim_env.agent.robot.get_qpos()[0].cpu().numpy()
    current_pos = sim_env.agent.tcp_pose.p[0].cpu().numpy()
    
    # 1. Cartesian interpolation for the approach
    steps = 40
    qpos_track = current_qpos.copy()
    
    active_mask = np.zeros(sim_env.agent.robot.dof, dtype=bool)
    active_mask[:7] = True  # Only IK the 7 arm joints
    
    for i in range(1, steps + 1):
        alpha = i / steps
        target_p = current_pos + alpha * (final_grasp_pos - current_pos)
        target_pose = sapien.Pose(p=target_p, q=final_grasp_quat)
        
        ik_qpos, success, error = model.compute_inverse_kinematics(
            link_idx,
            target_pose,
            initial_qpos=qpos_track,
            active_qmask=active_mask,
            max_iterations=100,
        )
        
        if success:
            qpos_track = ik_qpos
            
        # Stepping the environment requires a 7-dim action in our absolute action space
        action = qpos_track[:7]
        
        # Step the physics
        sim_env.step(action)
        if video_env is not None:
            video_env.step(action)
            frames.append(_frame_to_numpy(_render_video_frame(sim_env, video_env)))
            
    print("--- [Phase 1] Closing Gripper ---", flush=True)
    # 2. Close the passive gripper joints manually
    arm_names = [f"joint{i}" for i in range(1, 8)]
    gripper_indices = [
        i for i, name in enumerate(sim_env.agent.robot.get_active_joint_names()) 
        if name not in arm_names
    ]
    
    # We will step physics and manually increase the joint position of gripper joints
    close_steps = 30
    for i in range(1, close_steps + 1):
        # We incrementally set the qpos of the gripper joints. Max closed is roughly 0.85.
        target_val = 0.85 * (i / close_steps)
        current_qpos = sim_env.agent.robot.get_qpos()[0].cpu().numpy()
        for idx in gripper_indices:
            current_qpos[idx] = target_val
        
        # Teleport joints since they are passive/mimic and don't respond well to env.step()
        sim_env.agent.robot.set_qpos(current_qpos)
        if video_env is not None:
            video_env.agent.robot.set_qpos(current_qpos)
            
        # Keep the arm still
        action = current_qpos[:7]
        sim_env.step(action)
        if video_env is not None:
            video_env.step(action)
            frames.append(_frame_to_numpy(_render_video_frame(sim_env, video_env)))

    print("--- [Phase 1] Grasp Execution Complete ---\n", flush=True)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)

    # --- Load GraspGen ONCE before the episode loop ---
    if not _GRASPGEN_AVAILABLE:
        print(
            "ERROR: grasp_gen is not installed in this Python environment.\n"
            "Activate the GraspGen venv (e.g. source "
            "~/abhinav.pv/success/.venv/bin/activate) and re-run.",
            file=sys.stderr,
        )
        return 2
    graspgen_sampler = load_graspgen_sampler(args.graspgen_config)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(f"Failed to import ManiSkill/Gymnasium: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    register_pg3d_reach_envs()
    register_pg3d_xarm7_gripper_reach_envs()
    metadata = load_reach_metadata(args.dataset)
    if args.env_id_override is not None:
        metadata["env_id"] = args.env_id_override
    elif "XArm7" in str(metadata.get("env_id", "")):
        metadata["env_id"] = "PG3DReach-XArm7-RealObstacle-v0"
    else:
        metadata["env_id"] = "PG3DReach-RealObstacle-v0"

    # Override env_id to kitchen when requested
    if getattr(args, "kitchen_env", False):
        metadata["env_id"] = "PG3DReach-RealKitchen-v0"
        print(f"[graspgen_pick] env_id overridden to PG3DReach-RealKitchen-v0", flush=True)

    device = select_device(args.device)
    _seed_torch(args.seed)
    timer = TimingRecorder(
        enabled=args.profile,
        sync_fn=_cuda_sync_fn(device) if args.sync_cuda_timers else None,
    )
    policy = load_reach_policy_from_checkpoint(
        checkpoint_path,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)

    # Expand crop bounds for kitchen scene (same as pose steering script).
    new_bounds = crop_config.bounds.copy()
    new_bounds[0, 1] = max(new_bounds[0, 1], 0.7)
    new_bounds[2, 0] = 0.005
    crop_config = PointCloudCropConfig(
        bounds=new_bounds,
        num_points=crop_config.num_points,
        robot_point_fraction=0.25,
    )

    goal_thresh = (
        float(args.goal_thresh)
        if args.goal_thresh is not None
        else float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    )
    dataset_episode_seeds = [
        int(episode["seed"])
        for episode in metadata.get("episodes", [])
        if "seed" in episode
    ]
    zarr_root = (
        zarr.open_group(str(args.dataset), mode="r") if args.source == "dataset" else None
    )
    episode_indices = _episode_indices_from_args(
        args, dataset_episode_seeds=dataset_episode_seeds
    )
    specs = select_rollout_specs(
        source=args.source,
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=episode_indices,
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no episodes selected")

    # Store action_mode on args so _build_graspgen_constraint can read it.
    args.action_mode = action_mode

    artifact_seed = args.artifact_selection_seed
    video_episode_indices = (
        set(spec.output_index for spec in specs)
        if args.video
        else set(
            select_artifact_episode_indices(
                [spec.output_index for spec in specs],
                selection=args.artifact_selection,
                count=args.artifact_episode_count,
                seed=artifact_seed,
                every_episodes=args.video_every_episodes,
            )
        )
    )
    rerun_episode_indices = set(
        select_artifact_episode_indices(
            [spec.output_index for spec in specs],
            selection=args.artifact_selection,
            count=args.artifact_episode_count,
            seed=artifact_seed,
            every_episodes=args.rerun_every_episodes,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = _init_wandb(args, metadata=metadata, checkpoint_path=checkpoint_path)

    sim_env: Any | None = None
    ghost_env: Any | None = None
    rows: list[dict[str, Any]] = []
    metrics_path    = args.output_dir / "metrics.jsonl"
    decisions_path  = args.output_dir / "decisions.jsonl"
    step_traces_path = args.output_dir / "step_traces.jsonl"
    timings_path    = args.output_dir / "timings.jsonl"
    timing_written  = 0
    rng = np.random.default_rng(args.seed)

    try:
        sim_env = gym.make(
            str(metadata["env_id"]),
            **_env_kwargs(
                metadata,
                render_mode="rgb_array" if args.video else None,
                max_episode_steps=args.max_episode_steps,
            ),
        )
        ghost_env = gym.make(
            str(metadata["env_id"]),
            **_env_kwargs(metadata, render_mode=None,
                          max_episode_steps=args.max_episode_steps),
        )

        # ── Make marker spheres virtual ONLY in sim_env ─────────────────────
        # We completely strip the visual shapes from start_site and goal_site
        # in the sim_env so they NEVER appear in the point cloud cameras.
        # However, because ghost_env and video_env are separate env instances
        # (created via separate gym.make calls), they RETAIN their original
        # visual shapes. This perfectly matches the "virtual obstacle" pattern:
        # invisible to the policy, but visible in the video/debug rendering!
        _make_marker_spheres_strictly_virtual(sim_env)

        adapter = DP3ChunkPolicyAdapter(
            policy,
            action_mode=action_mode,
            device=device,
            policy_batch_size=args.policy_batch_size,
            timer=timer,
        )

        with (
            metrics_path.open("w", encoding="utf-8") as metrics_file,
            decisions_path.open("w", encoding="utf-8") as decisions_file,
            step_traces_path.open("w", encoding="utf-8") as step_traces_file,
        ):
            for spec in specs:
                zarr_context = (
                    _zarr_episode_context_with_paths(zarr_root, spec.dataset_episode_index)
                    if zarr_root is not None and spec.dataset_episode_index is not None
                    else None
                )

                # Disable zarr_context for jstbanana-v0 so run_eval_episode
                # doesn't forcefully reset the goal_site away from the object.
                is_banana = getattr(sim_env.unwrapped, "_JSTBANANA_Z", None) is not None
                if is_banana:
                    zarr_context = None

                # *** KEY DIFFERENCE FROM POSE STEERING: use GraspGen constraint ***
                constraints, pending_spawn = _build_graspgen_constraint(
                    sim_env,
                    spec=spec,
                    crop_config=crop_config,
                    graspgen_sampler=graspgen_sampler,
                    args=args,
                    zarr_context=zarr_context,
                )

                constraint_path = (
                    args.output_dir / "constraints" / f"episode_{spec.output_index:03d}.json"
                )
                with timer.time("json_write", artifact="constraint"):
                    save_episode_constraints(constraint_path, constraints)

                write_video = args.video and spec.output_index in video_episode_indices
                write_rerun = args.rerun and spec.output_index in rerun_episode_indices

                for method in args.methods:
                    # Expose current grasp data to the monkeypatched save_rerun_timeline
                    global CURRENT_RERUN_DATA
                    CURRENT_RERUN_DATA = getattr(args, "_graspgen_rerun_data", {}).get(spec.output_index)

                    row = run_eval_episode(
                        sim_env=sim_env,
                        ghost_env=ghost_env,
                        policy=policy,
                        adapter=adapter,
                        method=method,
                        spec=spec,
                        constraints=constraints,
                        pending_spawn=pending_spawn,
                        action_mode=action_mode,
                        goal_mask_radius=0.20,
                        crop_config=crop_config,
                        goal_thresh=goal_thresh,
                        output_dir=args.output_dir,
                        max_steps=args.max_steps,
                        post_success_steps=args.post_success_steps,
                        planning_horizon_chunks=args.planning_horizon_chunks,
                        execution_horizon_chunks=args.execution_horizon_chunks,
                        action_ema_alpha=args.action_ema_alpha,
                        geometry_mode=args.geometry_mode,
                        k_schedule=tuple(args.k_schedule),
                        gripper_open=args.gripper_open,
                        match_current_robot_points=args.match_current_robot_points,
                        video=write_video,
                        rerun=write_rerun,
                        video_fps=args.video_fps,
                        decisions_file=decisions_file,
                        step_traces_file=step_traces_file,
                        rng=rng,
                        timer=timer,
                        video_env_factory=_video_env_factory(
                            gym,
                            metadata=metadata,
                            enabled=write_video and args.constraint_overlay_video,
                            max_episode_steps=args.max_episode_steps,
                        ),
                        constraint_overlay_alpha=args.constraint_overlay_alpha,
                        constraint_overlay_color=tuple(args.constraint_overlay_color),
                        robot_clearance_metric=args.robot_clearance_metric,
                        robot_clearance_stride=args.robot_clearance_stride,
                        zarr_context=zarr_context,
                        parallel_pool=None,
                        post_episode_callback=_execute_open_loop_grasp,
                    )
                    rows.append(row)
                    with timer.time("json_write", artifact="metrics"):
                        metrics_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        metrics_file.flush()
                    _log_wandb_episode(run, args=args, row=row, global_step=len(rows))
                    print(
                        f"method={method} episode={spec.output_index} seed={spec.seed} "
                        f"combined={row['combined_success']} reach={row['reach_success']} "
                        f"constraint={row['constraint_satisfied']} "
                        f"final={_format_optional(row['final_target_distance'])} "
                        f"clearance={_format_optional(row.get('min_clearance'))}",
                        flush=True,
                    )
                    pm_list = row.get("cartesian_pose_metrics", [])
                    if pm_list:
                        pm = pm_list[0]
                        print("\n=== GRASP EXECUTION CHECK ===")
                        print(f"  Target Position:    {[round(x, 4) for x in pm['target_position']]}")
                        print(f"  Target Orientation: {[round(x, 4) for x in pm['target_orientation']]} (wxyz)")
                        if pm['min_position_error'] is not None:
                            print(f"  Achieved Pos Error: {pm['min_position_error']:.4f} m  (at step {pm['min_position_step']})")
                            print(f"  Achieved Rot Error: {pm['rotation_error_at_min_position']:.4f} rad (at best position)")
                        print(f"  Strictly Satisfied: {pm['satisfied']} (within {pm['position_tolerance']}m and {pm['rotation_tolerance']:.4f}rad)")
                        print("=============================\n", flush=True)
                timing_written = _write_new_timing_events(
                    timer, timings_path, start_index=timing_written
                )
                if should_emit_episode_artifact(spec.output_index, args.plot_every_episodes):
                    _maybe_emit_progress(
                        output_dir=args.output_dir,
                        rows=rows,
                        timer=timer,
                        episode_index=spec.output_index,
                        plots=args.plots or run is not None,
                        run=run,
                        args=args,
                    )

    except Exception as exc:
        print(f"Failed GraspGen pick eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1
    finally:
        if sim_env is not None:
            sim_env.close()
        if ghost_env is not None:
            ghost_env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(args.dataset),
        "source": args.source,
        "methods": list(args.methods),
        "env_id": metadata["env_id"],
        "constraint_source": _constraint_source_summary(args),
        "metrics_jsonl": str(metrics_path),
        "decisions_jsonl": str(decisions_path),
        "timing": timer.summary(),
        "episodes": rows,
        "by_method": summarize_metrics(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _print_timing_summary(timer)
    return 0


# ===========================================================================
#  run_eval_episode — imported from pose steering script's internal functions
#  via the shared pg3d.eval / pg3d.composition machinery.
#  We re-export it here so main() can call it cleanly.
# ===========================================================================

# run_eval_episode is too tightly coupled to the pose steering script to
# import directly, so we import the equivalent from pose steering via sys.path.
# This is the standard pattern used throughout the codebase.
def _import_run_eval_episode():
    import sys
    from pathlib import Path
    
    script_dir = str(Path(__file__).parent.resolve())
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    import eval_pointcloud_pose_steering_reach
    # Monkeypatch the module so that it uses the local save_rerun_timeline
    # which has the logic for logging GraspGen pitchforks into the main file.
    eval_pointcloud_pose_steering_reach.save_rerun_timeline = save_rerun_timeline
    return eval_pointcloud_pose_steering_reach.run_eval_episode


try:
    run_eval_episode = _import_run_eval_episode()
    # Monkeypatch the module's save_rerun_timeline so run_eval_episode calls our custom one!
    import eval_pointcloud_pose_steering_reach
    eval_pointcloud_pose_steering_reach.save_rerun_timeline = save_rerun_timeline
except Exception as e:
    _import_err_msg = str(e)
    def run_eval_episode(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "Could not import run_eval_episode from "
            "eval_pointcloud_pose_steering_reach.py: "
            f"{_import_err_msg}"
        )


# ===========================================================================
#  parse_args
# ===========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GraspGen-steered pick evaluation (Phase 0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset / checkpoint ---
    p.add_argument("--dataset", type=Path, required=True,
                   help="Path to the Zarr reach dataset.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to the DP3 reach checkpoint (.ckpt).")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Directory containing DP3 checkpoints (latest is used).")
    p.add_argument("--checkpoint-model", choices=["ema", "online"], default="ema")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for metrics, videos, and constraint JSON files.")
    p.add_argument("--env-id-override", type=str, default=None,
                   help="Override the env_id from dataset metadata.")
    p.add_argument("--kitchen-env", action="store_true",
                   help="Force use of PG3DReach-RealKitchen-v0.")

    # ---------------------------------------------------------------------------
    # *** GraspGen-specific args (NEW) ***
    # ---------------------------------------------------------------------------
    g = p.add_argument_group("GraspGen")
    g.add_argument("--graspgen-config", type=Path, required=True,
                   help="Path to the GraspGen gripper YAML config "
                        "(e.g. GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml).")
    g.add_argument("--graspgen-threshold", type=float, default=0.8,
                   help="Discriminator score threshold for keeping grasp candidates.")
    g.add_argument("--graspgen-num-grasps", type=int, default=200,
                   help="Number of diffusion grasp samples per inference call.")
    g.add_argument("--graspgen-z-offset", type=float, default=0.0,
                   help="Metres to shift the grasp contact point back along the approach "
                        "axis to account for gripper geometry difference. "
                        "0.0 for Robotiq 2F-140 (same as GraspGen training gripper).")
    g.add_argument("--grasp-approach-offset", type=float, default=0.10,
                   help="Distance (m) to offset the goal pose backwards along the approach "
                        "axis to create a pre-grasp pose. The DP3 reach policy will be "
                        "evaluated against this pre-grasp pose. (Phase 0 -> Phase 1)")
    g.add_argument("--grasp-object-crop-radius", type=float, default=0.10,
                   help="Sphere radius (m) around the object actor centroid for the GraspGen crop.")
    g.add_argument("--grasp-object-index", type=int, default=-1,
                   help="Legacy: index into env.unwrapped.ycb_objects[] to override the "
                        "actor-pose lookup. -1 = use automatic actor detection (preferred).")
    g.add_argument("--grasp-actor-name", type=str, default=None,
                   help="Named SAPIEN actor to use as the grasp target (optional, "
                        "overrides auto-detection for envs with multiple objects).")
    g.add_argument("--grasp-weight", type=float, default=2.0,
                   help="Weight of the CartesianPoseConstraint in candidate scoring.")
    g.add_argument("--grasp-position-tolerance", type=float, default=0.02,
                   help="Position tolerance (m) for CartesianPoseConstraint.satisfied().")
    g.add_argument("--grasp-rotation-tolerance", type=float, default=0.1745,
                   help="Rotation tolerance (rad) for CartesianPoseConstraint.satisfied(). 0.1745 rad = 10 degrees.")
    g.add_argument("--graspgen-viser", action="store_true",
                   help="Open a blocking Viser 3-D debug window after each GraspGen call, "
                        "showing the object cloud and all grasp candidates as pitchforks. "
                        "Close the browser tab or press Ctrl+C to continue.")

    # --- Episode / source ---
    p.add_argument("--source", choices=["dataset", "fresh"], default="fresh")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--episode-indices", type=int, nargs="+", default=None)
    p.add_argument("--episode-indices-file", type=Path, default=None)
    p.add_argument("--unique-dataset-seeds", action="store_true")

    # --- Eval methods ---
    p.add_argument("--methods", nargs="+",
                   choices=["base", "rejection", "reranking"],
                   default=["reranking"],
                   help="Evaluation methods to run.")

    # --- Planning ---
    p.add_argument("--planning-horizon-chunks", type=int, default=1)
    p.add_argument("--execution-horizon-chunks", type=int, default=1)
    p.add_argument("--k-schedule", type=int, nargs="+", default=[16])
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--post-success-steps", type=int, default=0)
    p.add_argument("--goal-thresh", type=float, default=None)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--action-ema-alpha", type=float, default=1.0)
    p.add_argument("--gripper-open", type=float, default=0.04,
                   help="Gripper open position (m). The reach ckpt does not predict "
                        "gripper actions; this value pads the sim action.")
    p.add_argument("--geometry-mode", choices=["fast", "exact"], default="exact",
                   help="World-model geometry mode. 'exact' uses the full robot point cloud "
                        "at each waypoint for collision scoring (slow but accurate). "
                        "'fast' uses a cached ghost approximation.")
    p.add_argument("--constraint-target", choices=["eef", "robot"], default="robot",
                   help="Which part of the robot to score against collision constraints. "
                        "'robot' uses the full robot point cloud; 'eef' uses only the "
                        "end-effector position. Default: robot.")
    p.add_argument("--match-current-robot-points", action="store_true")
    p.add_argument("--policy-batch-size", type=int, default=64)
    p.add_argument("--score-weights", type=float, nargs=4, default=None,
                   metavar=("GOAL", "SMOOTHNESS", "CONSTRAINT", "CONSENSUS"))

    # --- Posture (optional, stacked on top of GraspGen constraint) ---
    p.add_argument("--posture-target-joints", type=float, nargs="+", default=None)
    p.add_argument("--posture-weight", type=float, default=1.0)
    p.add_argument("--posture-eval-timestep", choices=["all", "final", "midpoint"],
                   default="all")

    # --- Video / Rerun ---
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument("--video-every-episodes", type=int, default=1)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--rerun-every-episodes", type=int, default=1)
    p.add_argument("--constraint-overlay-video", action="store_true")
    p.add_argument("--constraint-overlay-alpha", type=float, default=0.2)
    p.add_argument("--constraint-overlay-color", type=float, nargs=4,
                   default=[1.0, 0.0, 1.0, 0.2])

    # --- Artifact selection ---
    p.add_argument("--artifact-selection", choices=["periodic", "random", "all"], default="periodic")
    p.add_argument("--artifact-episode-count", type=int, default=3)
    p.add_argument("--artifact-selection-seed", type=int, default=0)
    p.add_argument("--plot-every-episodes", type=int, default=5)

    # --- Clearance metrics ---
    p.add_argument("--robot-clearance-metric", action="store_true")
    p.add_argument("--robot-clearance-stride", type=int, default=4)

    # --- Misc ---
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-every-episodes", type=int, default=10)
    p.add_argument("--sync-cuda-timers", action="store_true")
    p.add_argument("--wandb-project", type=str, default="pg3d-graspgen-pick")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["disabled", "online", "offline"], default="disabled")
    p.add_argument("--wandb-required", action="store_true")

    args = p.parse_args(argv)

    # Normalise output_dir
    args.output_dir = Path(args.output_dir)

    return args


if __name__ == "__main__":
    raise SystemExit(main())
