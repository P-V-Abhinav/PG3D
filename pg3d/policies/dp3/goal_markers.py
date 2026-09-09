from __future__ import annotations

from typing import Literal

import numpy as np

Array = np.ndarray

# Match the dataset bake (`--goal-marker-points 192 --goal-marker-radius 0.045`).
# The encoder splits off the trailing `goal_marker_points` slots as the goal
# branch, so this MUST equal the number of goal slots baked by the dataset
# writer or baked markers leak into the PointNet scene branch.
#
# The radius was 0.055 here while this comment said 0.045, and the artifact
# settles it: recovering the marker directly out of `pose_variety_final.zarr`
# (match the trailing 192 points of `data/point_cloud` against
# `goal_marker_offsets`) reproduces them at 0.000000 m error with
# shape="sphere", num_points=192, radius=0.045, on rows 0 / 226944 / 453888.
# Corrected to 0.045 on 2026-09-09. A 1 cm error here is not cosmetic: the
# encoder flattens the marker points, so every one of the 192 slots would have
# been offset radially from what the checkpoint was trained to read.
DEFAULT_GOAL_MARKER_POINTS = 192
DEFAULT_GOAL_MARKER_RADIUS = 0.045

MarkerShape = Literal["sphere", "cross_ring"]

# Which offset pattern new markers use. `sphere` is the shape every current
# artifact is baked with; `cross_ring` exists only to read artifacts baked
# before 2026-09-07. See `goal_marker_offsets` for why the choice is not
# cosmetic.
DEFAULT_GOAL_MARKER_SHAPE: MarkerShape = "sphere"


def _sphere_shell_offsets(num_points: int, r: np.float32) -> Array:
    """Fibonacci/golden-angle points on a sphere shell of radius ``r``."""
    indices = np.arange(num_points, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / float(num_points)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * golden_angle
    shell = np.stack([radial * np.cos(theta), radial * np.sin(theta), z], axis=1)
    return (r * shell).astype(np.float32)


def _cross_ring_offsets(num_points: int, r: np.float32) -> Array:
    """Legacy centre + axis-cross + stacked-ring pattern (pre-2026-09-07 bakes)."""
    pattern: list[np.ndarray] = [np.zeros(3, dtype=np.float32)]
    cross_dirs = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    for direction in cross_dirs:
        pattern.append(r * direction)

    ring_count = max(0, num_points - len(pattern))
    for idx in range(ring_count):
        angle = 2.0 * np.pi * idx / max(ring_count, 1)
        ring_radius = r * (0.70 if idx % 2 == 0 else 1.00)
        z_offset = r * 0.25 * (1.0 if idx % 4 in {0, 1} else -1.0)
        pattern.append(
            np.asarray(
                [ring_radius * np.cos(angle), ring_radius * np.sin(angle), z_offset],
                dtype=np.float32,
            )
        )

    if num_points <= len(pattern):
        return np.asarray(pattern[:num_points], dtype=np.float32)
    repeats = int(np.ceil(num_points / len(pattern)))
    return np.tile(np.asarray(pattern, dtype=np.float32), (repeats, 1))[:num_points]


def goal_marker_offsets(
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
    shape: MarkerShape = DEFAULT_GOAL_MARKER_SHAPE,
) -> Array:
    """Return deterministic offsets used for target-centered goal tokens.

    Placement is a pure function of ``num_points``/``radius``/``shape`` -- no
    RNG -- so ``offsets[k]`` means the same thing on every call. That
    determinism is load-bearing, not just reproducible: these offsets feed
    ``PointNetEncoderXYZ.goal_marker_mlp`` (pg3d/policies/dp3/modules.py),
    which *flattens* the marker points into one fixed-length vector rather
    than pooling them order-invariantly. Index ``k`` has to be the same offset
    direction every call, or the MLP computes a different function of the same
    target position -- and a policy trained under one shape reads a goal that
    is quietly wrong under another, with nothing raising.

    ``sphere`` (default) places every point exactly on the shell at ``radius``
    via a golden-angle spiral. A sphere is SO(3)-symmetric, so the marker
    carries no orientation information by construction.

    ``cross_ring`` is the pre-2026-09-07 pattern: a centre point, six axis
    points, then stacked rings at 0.70/1.00 of ``radius``. Retained only so
    artifacts baked with it stay readable. It differs from ``sphere`` by up to
    ~2x ``radius`` per index, which is why the two are not interchangeable --
    see ADR 0022.
    """
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if shape not in ("sphere", "cross_ring"):
        raise ValueError(f"unknown goal marker shape {shape!r}")
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    r = np.float32(radius)
    if r == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    if shape == "sphere":
        return _sphere_shell_offsets(num_points, r)
    return _cross_ring_offsets(num_points, r)


def goal_marker_points(
    target_position: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
    shape: MarkerShape = DEFAULT_GOAL_MARKER_SHAPE,
) -> Array:
    """Return fixed ordered marker points centered at each target position."""
    target = np.asarray(target_position, dtype=np.float32)
    if target.shape[-1:] != (3,):
        raise ValueError(f"target_position must end with shape [3], got {target.shape}")
    offsets = goal_marker_offsets(num_points=num_points, radius=radius, shape=shape)
    if num_points == 0:
        return np.zeros((*target.shape[:-1], 0, 3), dtype=np.float32)
    return target[..., None, :] + offsets.reshape((1,) * (target.ndim - 1) + offsets.shape)


def insert_goal_marker_points(
    point_cloud: Array,
    target_position: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
    shape: MarkerShape = DEFAULT_GOAL_MARKER_SHAPE,
) -> Array:
    """Overwrite the final ``num_points`` point-cloud slots with ordered goal tokens."""
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.shape[-1:] != (3,):
        raise ValueError(f"point_cloud must end with shape [*, 3], got {points.shape}")
    if points.ndim < 2:
        raise ValueError(f"point_cloud must have at least 2 dimensions, got {points.shape}")
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if num_points == 0:
        return points.astype(np.float32, copy=True)
    if num_points >= points.shape[-2]:
        raise ValueError(
            "num_points must be smaller than the point-cloud point count "
            f"({num_points} >= {points.shape[-2]})"
        )

    marker = goal_marker_points(target_position, num_points=num_points, radius=radius, shape=shape)
    expected_marker_shape = (*points.shape[:-2], num_points, 3)
    try:
        marker = np.broadcast_to(marker, expected_marker_shape)
    except ValueError as exc:
        raise ValueError(
            f"target_position shape {np.asarray(target_position).shape} cannot broadcast "
            f"to point_cloud shape {points.shape}"
        ) from exc

    output = points.astype(np.float32, copy=True)
    output[..., -num_points:, :] = marker
    return output
