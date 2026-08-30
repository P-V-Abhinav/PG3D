"""Parity test: dataset-writer and inference-time triad marker baking agree.

Both the dataset writer (`_build_goal_marker`) and the inference-time bakers
(`insert_goal_marker_points` in reach_dataset / rollout) route through the same
shared `build_goal_marker` in goal_markers.py. This test locks that parity: for
the same (position, quaternion, style, num_points, radius), the trailing goal
slots are identical regardless of which entry point produced them.
"""

from __future__ import annotations

import numpy as np

from pg3d.policies.dp3.goal_markers import (
    build_goal_marker,
    insert_goal_marker_points,
)


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))], dtype=np.float32)


def test_writer_and_inference_triad_marker_parity():
    target = np.array([0.25, -0.1, 0.42], dtype=np.float32)
    quat = _quat_axis_angle([0.2, 0.9, 0.3], 0.9)
    num_points = 48
    radius = 0.05

    # Writer path: build the marker directly (as _inject_point_cloud_saliency does).
    writer_marker = build_goal_marker(
        target, quat, style="triad", num_points=num_points, radius=radius
    ).reshape(num_points, 3)

    # Inference path: insert into a point cloud (as reach_dataset/__getitem__ does),
    # then read back the trailing goal slots.
    pc = np.zeros((128, 3), dtype=np.float32)
    baked = insert_goal_marker_points(
        pc, target, num_points=num_points, radius=radius, style="triad", quat=quat
    )
    inference_marker = baked[-num_points:]

    np.testing.assert_allclose(writer_marker, inference_marker, atol=1e-6)


def test_per_row_quat_matches_single_quat_when_uniform():
    # A per-row quaternion array of identical rows must match the single-quat path.
    target = np.tile(np.array([0.3, 0.0, 0.4], dtype=np.float32), (5, 1))
    quat = _quat_axis_angle([1.0, 0.0, 0.0], np.pi / 4)
    single = build_goal_marker(target, quat, style="triad", num_points=32, radius=0.05)
    per_row = build_goal_marker(
        target,
        np.tile(quat, (5, 1)),
        style="triad",
        num_points=32,
        radius=0.05,
    )
    np.testing.assert_allclose(single, per_row, atol=1e-6)
