"""Tests for the new diagnose_reach_dataset diagnostics (pose multimodality + coverage)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from pg3d.envs.maniskill_adapter.dataset import ReachEpisodeData, write_reach_zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnose_reach_dataset.py"


def _load_diag_module():
    spec = importlib.util.spec_from_file_location("_diag_mod", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"diagnose module import failed: {exc}")
    return module


def _quat_downward_tilt(theta_deg, phi_deg=0.0):
    from scipy.spatial.transform import Rotation

    base = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
    tr = np.radians(theta_deg)
    pr = np.radians(phi_deg)
    tilt = Rotation.from_rotvec(np.array([np.cos(pr), np.sin(pr), 0.0]) * tr)
    q = (tilt * base).as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def test_pose_multimodality_stats_counts_configs():
    module = _load_diag_module()
    metadata = {
        "episodes": [
            {"seed": 1, "orientation_mode": "downward", "nullspace_config_id": 0},
            {"seed": 1, "orientation_mode": "downward", "nullspace_config_id": 1},
            {"seed": 1, "orientation_mode": "downward", "nullspace_config_id": 1},  # dup id
            {"seed": 1, "orientation_mode": "cone_001", "nullspace_config_id": 0},
        ]
    }
    stats = module._pose_multimodality_stats(metadata)
    assert stats["available"] is True
    assert stats["num_pose_groups"] == 2
    # downward group has 2 distinct configs, cone_001 has 1.
    assert stats["configs_per_pose"]["max"] == 2
    assert stats["configs_per_pose"]["min"] == 1


def test_pose_multimodality_stats_absent_when_no_field():
    module = _load_diag_module()
    stats = module._pose_multimodality_stats({"episodes": [{"seed": 1}]})
    assert stats["available"] is False


def test_orientation_coverage_from_goal_quat(tmp_path):
    module = _load_diag_module()
    # Build a tiny zarr with a goal_quat array spanning several tilt angles.
    episode_length = 2
    tilts = [0.0, 20.0, 40.0, 55.0]
    episodes = []
    for i in range(len(tilts)):
        n = episode_length
        episodes.append(
            ReachEpisodeData(
                state=np.zeros((n, 9), dtype=np.float32),
                action=np.zeros((n, 7), dtype=np.float32),
                sim_action=np.zeros((n, 8), dtype=np.float32),
                point_cloud=np.zeros((n, 8, 3), dtype=np.float32),
                robot_mask=np.zeros((n, 8), dtype=bool),
                point_valid_mask=np.ones((n, 8), dtype=bool),
                target_position=np.zeros((n, 3), dtype=np.float32),
                tcp_pose=np.zeros((n, 7), dtype=np.float32),
                success=np.ones((n,), dtype=bool),
                metadata={"seed": i, "success": True},
            )
        )
    out = tmp_path / "reach.zarr"
    write_reach_zarr(
        out, episodes, metadata={"env_id": "X", "env_kwargs": {}, "action_mode": "abs_joint"}
    )
    root = zarr.open_group(str(out), mode="a")
    data = root["data"]
    per_step = np.concatenate(
        [np.tile(_quat_downward_tilt(t), (episode_length, 1)) for t in tilts], axis=0
    ).astype(np.float32)
    data.array(name="goal_quat", data=per_step, chunks=(per_step.shape[0], 4))

    metadata = {"orientation_sampling": {"cone_half_angle_deg": 60.0}}
    stats = module._orientation_coverage_stats(root["data"], metadata)
    assert stats["available"] is True
    assert stats["num_unique_orientations"] == 4
    assert stats["tilt_deg"]["max"] == pytest.approx(55.0, abs=1.0)
    assert stats["tilt_deg"]["min"] == pytest.approx(0.0, abs=1e-3)


def test_orientation_coverage_absent_without_goal_quat(tmp_path):
    module = _load_diag_module()
    episodes = [
        ReachEpisodeData(
            state=np.zeros((2, 9), dtype=np.float32),
            action=np.zeros((2, 7), dtype=np.float32),
            sim_action=np.zeros((2, 8), dtype=np.float32),
            point_cloud=np.zeros((2, 8, 3), dtype=np.float32),
            robot_mask=np.zeros((2, 8), dtype=bool),
            point_valid_mask=np.ones((2, 8), dtype=bool),
            target_position=np.zeros((2, 3), dtype=np.float32),
            tcp_pose=np.zeros((2, 7), dtype=np.float32),
            success=np.ones((2,), dtype=bool),
            metadata={"seed": 0, "success": True},
        )
    ]
    out = tmp_path / "reach_noquat.zarr"
    write_reach_zarr(
        out, episodes, metadata={"env_id": "X", "env_kwargs": {}, "action_mode": "abs_joint"}
    )
    root = zarr.open_group(str(out), mode="r")
    stats = module._orientation_coverage_stats(root["data"], {})
    assert stats["available"] is False
