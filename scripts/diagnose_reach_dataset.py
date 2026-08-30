from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata
from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_POINTS,
    DEFAULT_GOAL_MARKER_RADIUS,
    goal_marker_points,
    insert_goal_marker_points,
)
from pg3d.policies.dp3.reach_dataset import validation_episode_mask
from pg3d.utils.serialization import jsonable


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = load_reach_metadata(args.dataset)
    root = zarr.open_group(str(args.dataset), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    first_indices = _episode_first_indices(episode_ends)
    target_first = np.asarray(data["target_position"][first_indices], dtype=np.float32)
    labels = _region_labels(target_first, metadata)

    sampled_rows = _sample_rows(int(data["point_cloud"].shape[0]), max_rows=args.max_rows)
    raw_points = np.asarray(data["point_cloud"][sampled_rows], dtype=np.float32)
    targets = np.asarray(data["target_position"][sampled_rows], dtype=np.float32)
    robot_mask = (
        np.asarray(data["robot_mask"][sampled_rows], dtype=bool)
        if "robot_mask" in data
        else np.zeros(raw_points.shape[:2], dtype=bool)
    )
    valid_mask = (
        np.asarray(data["point_valid_mask"][sampled_rows], dtype=bool)
        if "point_valid_mask" in data
        else np.ones(raw_points.shape[:2], dtype=bool)
    )
    transformed = insert_goal_marker_points(
        raw_points,
        targets,
        num_points=args.goal_marker_points,
        radius=args.goal_marker_radius,
    )
    expected_markers = goal_marker_points(
        targets,
        num_points=args.goal_marker_points,
        radius=args.goal_marker_radius,
    )
    marker_error = (
        float(np.max(np.abs(transformed[:, -args.goal_marker_points :] - expected_markers)))
        if args.goal_marker_points
        else 0.0
    )

    val_mask = validation_episode_mask(
        len(episode_ends),
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    train_mask = ~val_mask
    summary = {
        "dataset": str(args.dataset),
        "env_id": metadata.get("env_id"),
        "num_episodes": int(len(episode_ends)),
        "num_steps": int(episode_ends[-1]) if len(episode_ends) else 0,
        "goal_marker": {
            "points": args.goal_marker_points,
            "radius": args.goal_marker_radius,
            "max_marker_error": marker_error,
        },
        "target_first_obs": _axis_summary(target_first),
        "regions": {
            "all": _label_counts(labels),
            "train": _label_counts(labels[train_mask]),
            "val": _label_counts(labels[val_mask]),
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
        },
        "raw_goal_visibility": _goal_visibility(
            raw_points,
            targets,
            robot_mask=robot_mask,
            valid_mask=valid_mask,
            thresholds=(0.04, 0.10, 0.20),
        ),
        "pose_multimodality": _pose_multimodality_stats(metadata),
        "orientation_coverage": _orientation_coverage_stats(data, metadata),
    }
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report target distribution and P11 goal-marker diagnostics for a reach Zarr."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--goal-marker-points", type=int, default=DEFAULT_GOAL_MARKER_POINTS)
    parser.add_argument("--goal-marker-radius", type=float, default=DEFAULT_GOAL_MARKER_RADIUS)
    args = parser.parse_args(argv)
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.goal_marker_points < 0:
        raise ValueError("--goal-marker-points must be non-negative")
    if args.goal_marker_radius < 0:
        raise ValueError("--goal-marker-radius must be non-negative")
    return args


def _episode_first_indices(episode_ends: np.ndarray) -> np.ndarray:
    if len(episode_ends) == 0:
        return np.zeros((0,), dtype=np.int64)
    starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    return starts.astype(np.int64, copy=False)


def _sample_rows(num_rows: int, *, max_rows: int) -> np.ndarray:
    if num_rows <= max_rows:
        return np.arange(num_rows, dtype=np.int64)
    return np.linspace(0, num_rows - 1, max_rows).round().astype(np.int64)


def _region_labels(targets: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    task = metadata.get("task", {})
    regions = task.get("goal_regions", []) if isinstance(task, dict) else []
    if not regions:
        return np.full((targets.shape[0],), "unclassified", dtype=object)
    core = next((region for region in regions if region.get("name") == "core_practical"), None)
    bounds = np.asarray(task.get("goal_bounds"), dtype=np.float32)
    labels = np.full((targets.shape[0],), "outside", dtype=object)
    if bounds.shape == (3, 2):
        in_union = np.all((targets >= bounds[:, 0]) & (targets <= bounds[:, 1]), axis=1)
        labels[in_union] = "outer_practical"
    if core is not None:
        core_bounds = np.asarray(core.get("bounds"), dtype=np.float32)
        if core_bounds.shape == (3, 2):
            in_core = np.all(
                (targets >= core_bounds[:, 0]) & (targets <= core_bounds[:, 1]),
                axis=1,
            )
            labels[in_core] = "core_practical"
    return labels


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels.astype(str), return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts, strict=True)}


def _axis_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"min": [0.0, 0.0, 0.0], "mean": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def _goal_visibility(
    point_cloud: np.ndarray,
    targets: np.ndarray,
    *,
    robot_mask: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    distances = np.linalg.norm(point_cloud - targets[:, None, :], axis=-1)
    invalid = ~valid_mask
    nearest_all = np.min(np.where(invalid, np.inf, distances), axis=1)
    nearest_nonrobot = np.min(np.where(invalid | robot_mask, np.inf, distances), axis=1)
    return {
        "sampled_rows": int(point_cloud.shape[0]),
        "nearest_all": _finite_summary(nearest_all),
        "nearest_nonrobot": _finite_summary(nearest_nonrobot),
        "fraction_without_nonrobot_point_within": {
            f"{threshold:.2f}m": float(np.mean(nearest_nonrobot > threshold))
            for threshold in thresholds
        },
    }


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def _pose_multimodality_stats(metadata: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-pose null-space config counts from episode metadata.

    Groups episodes by (seed, orientation_mode) and counts the distinct
    ``nullspace_config_id`` values seen per group. This measures how many distinct
    arm configurations were generated for each fixed 6D goal pose.
    """
    episodes = metadata.get("episodes", [])
    if not episodes:
        return {"available": False, "note": "no per-episode metadata"}
    groups: dict[tuple, set] = {}
    has_config_field = False
    for ep in episodes:
        if "nullspace_config_id" in ep:
            has_config_field = True
        seed = ep.get("seed")
        ori = ep.get("orientation_mode", "unknown")
        cfg = int(ep.get("nullspace_config_id", 0))
        groups.setdefault((seed, ori), set()).add(cfg)
    if not has_config_field:
        return {
            "available": False,
            "note": "dataset predates nullspace_config_id (no pose multimodality metadata)",
        }
    counts = np.asarray([len(v) for v in groups.values()], dtype=np.int64)
    hist: dict[str, int] = {}
    for c in counts.tolist():
        key = str(int(c))
        hist[key] = hist.get(key, 0) + 1
    return {
        "available": True,
        "num_pose_groups": int(len(groups)),
        "configs_per_pose": {
            "min": int(counts.min()) if counts.size else 0,
            "mean": float(counts.mean()) if counts.size else 0.0,
            "max": int(counts.max()) if counts.size else 0,
        },
        "config_count_histogram": hist,
    }


def _orientation_coverage_stats(data: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Equal-area cap-cell coverage histogram of feasible goal orientations.

    Reads the per-step ``goal_quat`` array (constant within an episode), computes
    the tilt angle of the gripper approach axis away from straight-down, and bins
    the achieved (feasible) orientations into equal-area cap cells (uniform in
    ``1 - cos(theta)``). This reports how much of the down-facing cone the dataset
    actually covers, not just what was attempted.
    """
    if "goal_quat" not in data:
        return {
            "available": False,
            "note": "dataset has no goal_quat array (position-only / sphere marker)",
        }
    from pg3d.policies.dp3.goal_markers import _quat_to_rotation_matrix

    quats = np.asarray(data["goal_quat"][:], dtype=np.float64)
    if quats.ndim != 2 or quats.shape[1] != 4 or quats.shape[0] == 0:
        return {"available": False, "note": "goal_quat has unexpected shape"}
    # Deduplicate to per-episode orientations to avoid weighting by episode length.
    unique_quats = np.unique(np.round(quats, 6), axis=0)
    down = np.array([0.0, 0.0, -1.0])
    tilt_deg = []
    for q in unique_quats:
        try:
            R = _quat_to_rotation_matrix(q)
        except ValueError:
            continue
        axis = R @ np.array([0.0, 0.0, 1.0])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        cos_t = float(np.clip(np.dot(axis, down), -1.0, 1.0))
        tilt_deg.append(np.degrees(np.arccos(cos_t)))
    tilt_deg = np.asarray(tilt_deg, dtype=np.float64)
    if tilt_deg.size == 0:
        return {"available": False, "note": "no valid goal quaternions"}

    pose_mm = metadata.get("orientation_sampling", {})
    half_angle = float(pose_mm.get("cone_half_angle_deg", max(60.0, float(tilt_deg.max()))))
    half_angle = max(half_angle, float(tilt_deg.max()), 1e-6)
    # Equal-area cap cells: uniform bins in (1 - cos(theta)) over [0, 1 - cos(half)].
    n_bins = 6
    cos_half = np.cos(np.radians(half_angle))
    u = 1.0 - np.cos(np.radians(tilt_deg))
    u_max = max(1.0 - cos_half, 1e-9)
    bin_idx = np.clip((u / u_max * n_bins).astype(int), 0, n_bins - 1)
    hist = {str(i): int(np.sum(bin_idx == i)) for i in range(n_bins)}
    filled = sum(1 for v in hist.values() if v > 0)
    return {
        "available": True,
        "cone_half_angle_deg": half_angle,
        "num_unique_orientations": int(tilt_deg.size),
        "tilt_deg": {
            "min": float(tilt_deg.min()),
            "mean": float(tilt_deg.mean()),
            "max": float(tilt_deg.max()),
        },
        "equal_area_cap_cell_counts": hist,
        "cap_cells_filled": f"{filled}/{n_bins}",
        "note": (
            "Cells are equal-area bands of the spherical cap (uniform in 1-cos(theta)); "
            "an even fill indicates unbiased cone coverage."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
