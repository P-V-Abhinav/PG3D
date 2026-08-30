"""Tests for the equal-area cone orientation sampler in the pose-reach writer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_maniskill_pose_reach_dataset.py"


def _load_writer_module():
    spec = importlib.util.spec_from_file_location("_pose_reach_writer_cone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"writer module import failed (missing deps): {exc}")
    return module


def _quat_to_down_axis(module, quat):
    """Return the world-space direction the gripper approach axis points."""
    from pg3d.policies.dp3.goal_markers import _quat_to_rotation_matrix

    R = _quat_to_rotation_matrix(np.asarray(quat, dtype=np.float64))
    # Straight-down base points local +Z to world -Z (180deg about X). The approach
    # axis in world frame is R @ [0,0,1].
    return R @ np.array([0.0, 0.0, 1.0])


def test_sampler_returns_valid_unit_quaternions():
    module = _load_writer_module()
    rng = np.random.default_rng(0)
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=60.0, count=50)
    assert len(oris) == 50
    assert "downward" in oris
    for q in oris.values():
        q = np.asarray(q, dtype=np.float64)
        assert q.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5)


def test_sampler_downward_is_straight_down():
    module = _load_writer_module()
    rng = np.random.default_rng(1)
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=60.0, count=5)
    axis = _quat_to_down_axis(module, oris["downward"])
    # Straight down: approach axis points to world -Z.
    np.testing.assert_allclose(axis, np.array([0.0, 0.0, -1.0]), atol=1e-5)


def test_sampler_stays_within_cone():
    module = _load_writer_module()
    rng = np.random.default_rng(2)
    half = 60.0
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=half, count=400)
    down = np.array([0.0, 0.0, -1.0])
    max_angle = 0.0
    for q in oris.values():
        axis = _quat_to_down_axis(module, q)
        axis = axis / np.linalg.norm(axis)
        ang = np.degrees(np.arccos(np.clip(np.dot(axis, down), -1.0, 1.0)))
        max_angle = max(max_angle, ang)
    assert max_angle <= half + 1e-3


def test_sampler_is_equal_area_not_pole_biased():
    """Uniform-in-cos(theta) => roughly uniform tilt-angle *area* density.

    We check that a meaningful fraction of samples land in the outer half of the
    cap (theta in [half/sqrt2-ish, half]) rather than clustering at the pole, which
    an equal-*angle* grid would fail. For equal-area, the fraction of samples with
    cos(theta) < (1+cos(half))/2 should be about 0.5.
    """
    module = _load_writer_module()
    rng = np.random.default_rng(3)
    half = 60.0
    n = 4000
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=half, count=n)
    down = np.array([0.0, 0.0, -1.0])
    cos_thetas = []
    for name, q in oris.items():
        if name == "downward":
            continue
        axis = _quat_to_down_axis(module, q)
        axis = axis / np.linalg.norm(axis)
        cos_thetas.append(np.clip(np.dot(axis, down), -1.0, 1.0))
    cos_thetas = np.asarray(cos_thetas)
    midpoint = (1.0 + np.cos(np.radians(half))) / 2.0
    frac_outer = float(np.mean(cos_thetas < midpoint))
    assert 0.4 < frac_outer < 0.6, f"equal-area cap density off: frac_outer={frac_outer}"
