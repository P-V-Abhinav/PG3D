"""Writer-side goal-marker saliency baking tests (no simulator required).

These import the standalone writer module directly and exercise the pure-numpy
saliency injection path to confirm the triad marker is orientation-sensitive and
the sphere marker is not, matching the shared goal_markers builder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_maniskill_pose_reach_dataset.py"


def _load_writer_module():
    spec = importlib.util.spec_from_file_location("_pose_reach_writer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - only when heavy deps missing
        pytest.skip(f"writer module import failed (missing deps): {exc}")
    return module


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))], dtype=np.float32)


def _make_row(num_points=128):
    rng = np.random.default_rng(0)
    return {
        "point_cloud": rng.standard_normal((num_points, 3)).astype(np.float32) * 0.01
        + np.array([0.3, 0.0, 0.4], dtype=np.float32),
        "point_valid_mask": np.ones(num_points, dtype=bool),
        "robot_mask": np.zeros(num_points, dtype=bool),
        "target_position": np.array([0.3, 0.0, 0.4], dtype=np.float32),
        "tcp_pose": np.array([0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }


def test_writer_triad_is_orientation_sensitive_sphere_is_not():
    module = _load_writer_module()
    cfg_triad = module.PointCloudSaliencyConfig(
        goal_marker_points=48,
        goal_marker_radius=0.05,
        tcp_marker_points=0,
        goal_marker_style="triad",
    )
    cfg_sphere = module.PointCloudSaliencyConfig(
        goal_marker_points=48,
        goal_marker_radius=0.05,
        tcp_marker_points=0,
        goal_marker_style="sphere",
    )
    bounds = np.array([[-2, 2], [-2, 2], [-2, 2]], dtype=np.float32)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q2 = _quat_axis_angle([0, 1, 0], np.pi / 2)

    # Sphere: goal slots identical regardless of quaternion.
    r1 = _make_row()
    r2 = _make_row()
    module._inject_point_cloud_saliency(
        r1, saliency_config=cfg_sphere, crop_bounds=bounds, goal_quat=q1
    )
    module._inject_point_cloud_saliency(
        r2, saliency_config=cfg_sphere, crop_bounds=bounds, goal_quat=q2
    )
    np.testing.assert_allclose(r1["point_cloud"][-48:], r2["point_cloud"][-48:])

    # Triad: goal slots differ with quaternion.
    t1 = _make_row()
    t2 = _make_row()
    module._inject_point_cloud_saliency(
        t1, saliency_config=cfg_triad, crop_bounds=bounds, goal_quat=q1
    )
    module._inject_point_cloud_saliency(
        t2, saliency_config=cfg_triad, crop_bounds=bounds, goal_quat=q2
    )
    assert not np.allclose(t1["point_cloud"][-48:], t2["point_cloud"][-48:], atol=1e-4)


def test_writer_triad_falls_back_to_sphere_without_quat():
    module = _load_writer_module()
    cfg = module.PointCloudSaliencyConfig(
        goal_marker_points=48,
        goal_marker_radius=0.05,
        tcp_marker_points=0,
        goal_marker_style="triad",
    )
    bounds = np.array([[-2, 2], [-2, 2], [-2, 2]], dtype=np.float32)
    row = _make_row()
    # Should not raise when goal_quat is None (fallback to sphere).
    module._inject_point_cloud_saliency(
        row, saliency_config=cfg, crop_bounds=bounds, goal_quat=None
    )
    assert row["point_valid_mask"][-48:].all()
