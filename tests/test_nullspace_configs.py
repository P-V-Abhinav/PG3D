"""Unit tests for the null-space config sampler (fake planner, no simulator)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_maniskill_pose_reach_dataset.py"


def _load_writer_module():
    spec = importlib.util.spec_from_file_location("_pose_reach_writer_ns", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"writer module import failed (missing deps): {exc}")
    return module


@dataclass
class _FakePose:
    p: np.ndarray
    q: np.ndarray


GOAL_XYZ = np.array([0.3, 0.0, 0.4], dtype=np.float64)
GOAL_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # SAPIEN w,x,y,z


def test_dedup_keeps_only_distinct_configs():
    module = _load_writer_module()
    cands = [
        {"qpos": np.zeros(7)},
        {"qpos": np.array([0.05, 0, 0, 0, 0, 0, 0.0])},  # near-duplicate of first
        {"qpos": np.array([1.0, 0, 0, 0, 0, 0, 0.0])},  # distinct
        {"qpos": np.array([1.02, 0, 0, 0, 0, 0, 0.0])},  # near-duplicate of third
    ]
    kept = module._dedup_configs_by_joint_distance(cands, min_joint_sep_rad=0.3)
    assert len(kept) == 2
    np.testing.assert_allclose(kept[0]["qpos"], np.zeros(7))
    np.testing.assert_allclose(kept[1]["qpos"][0], 1.0, atol=1e-6)


def test_quat_angular_distance():
    module = _load_writer_module()
    q0 = np.array([1.0, 0, 0, 0.0])
    # 90deg about X in SAPIEN [w,x,y,z]
    q90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0.0])
    assert abs(module._quat_angular_distance_deg(q0, q0)) < 1e-6
    assert abs(module._quat_angular_distance_deg(q0, q90) - 90.0) < 1e-3


def test_sampler_finds_distinct_configs_from_fake_planner():
    module = _load_writer_module()
    goal = _FakePose(GOAL_XYZ.copy(), GOAL_QUAT.copy())

    # Fake IK: three distinct on-tolerance branches + one off-tolerance branch.
    # The planner maps a jittered seed to the nearest "branch" config.
    branches = [
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        np.array([1.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        np.array([-1.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    ]
    bad_branch = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # lands off the goal

    def plan_fn(seed_qpos):
        seed0 = float(seed_qpos[0])
        # Route seeds to a branch by first-joint sign/magnitude; some seeds go bad.
        if seed0 > 1.5:
            return bad_branch
        pool = branches + [bad_branch]
        # nearest branch by first joint
        return min(pool, key=lambda b: abs(b[0] - seed0))

    def tcp_of_qpos(qpos):
        qpos = np.asarray(qpos)
        if np.allclose(qpos, bad_branch):
            # Off-tolerance: 10cm away and 30deg rotated.
            return np.array([0.4, 0.0, 0.4, np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0.0])
        # On-tolerance branches map exactly to the goal pose.
        return np.concatenate([GOAL_XYZ, GOAL_QUAT])

    rng = np.random.default_rng(0)
    configs = module.sample_nullspace_configs(
        goal_pose=goal,
        start_qpos=np.zeros(7),
        num_configs=3,
        pos_tol_m=0.01,
        rot_tol_deg=3.0,
        min_joint_sep_rad=0.5,
        rng=rng,
        seed_jitter_rad=1.5,
        max_attempts=200,
        plan_fn=plan_fn,
        tcp_of_qpos_fn=tcp_of_qpos,
    )
    # Should recover the distinct on-tolerance branches and never the bad one.
    assert 2 <= len(configs) <= 3
    for cfg in configs:
        assert cfg["position_error_m"] <= 0.01
        assert cfg["orientation_error_deg"] <= 3.0
        assert not np.allclose(cfg["qpos"], bad_branch)
    # Accepted configs are mutually distinct.
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            d = np.linalg.norm(np.asarray(configs[i]["qpos"]) - np.asarray(configs[j]["qpos"]))
            assert d >= 0.5


def test_sampler_rejects_all_when_out_of_tolerance():
    module = _load_writer_module()
    goal = _FakePose(GOAL_XYZ.copy(), GOAL_QUAT.copy())

    def plan_fn(seed_qpos):
        return np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    def tcp_of_qpos(qpos):
        return np.array([0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])  # far from goal

    rng = np.random.default_rng(1)
    configs = module.sample_nullspace_configs(
        goal_pose=goal,
        start_qpos=np.zeros(7),
        num_configs=3,
        pos_tol_m=0.01,
        rot_tol_deg=3.0,
        min_joint_sep_rad=0.5,
        rng=rng,
        plan_fn=plan_fn,
        tcp_of_qpos_fn=tcp_of_qpos,
    )
    assert configs == []
