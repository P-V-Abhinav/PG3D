"""End-to-end: ReachSequenceDataset bakes an orientation-sensitive triad marker.

Builds a tiny reach zarr, attaches a per-step ``goal_quat`` array (as the writer's
``_ensure_goal_observation_aliases`` does), and checks that the triad-configured
dataset produces different goal slots for episodes with different goal
orientations, while the sphere config does not.
"""

from __future__ import annotations

import numpy as np
import zarr

from pg3d.envs.maniskill_adapter.dataset import ReachEpisodeData, write_reach_zarr
from pg3d.policies.dp3.reach_dataset import ReachDatasetConfig, ReachSequenceDataset


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))], dtype=np.float32)


def _build_zarr_with_goal_quat(tmp_path, quats):
    episode_length = 4
    num_points = 32
    episodes = []
    for episode_idx in range(len(quats)):
        state = np.full((episode_length, 9), episode_idx, dtype=np.float32)
        action = state[:, :7] + 0.1
        point_cloud = np.zeros((episode_length, num_points, 3), dtype=np.float32)
        episodes.append(
            ReachEpisodeData(
                state=state,
                action=action.astype(np.float32),
                sim_action=np.concatenate(
                    [action, np.zeros((episode_length, 1), dtype=np.float32)], axis=1
                ),
                point_cloud=point_cloud,
                robot_mask=np.zeros((episode_length, num_points), dtype=bool),
                point_valid_mask=np.ones((episode_length, num_points), dtype=bool),
                target_position=np.tile(
                    np.array([0.3, 0.0, 0.4], dtype=np.float32), (episode_length, 1)
                ),
                tcp_pose=np.zeros((episode_length, 7), dtype=np.float32),
                success=np.ones((episode_length,), dtype=bool),
                metadata={"seed": episode_idx, "success": True},
            )
        )
    output = tmp_path / "reach.zarr"
    write_reach_zarr(
        output,
        episodes,
        metadata={"env_id": "PG3DReach-Narrow-v0", "env_kwargs": {}, "action_mode": "abs_joint"},
    )
    # Attach a per-step goal_quat array (episode-constant), like the writer alias.
    root = zarr.open_group(str(output), mode="a")
    data = root["data"]
    per_step = np.concatenate([np.tile(q, (episode_length, 1)) for q in quats], axis=0).astype(
        np.float32
    )
    data.array(name="goal_quat", data=per_step, chunks=(per_step.shape[0], 4))
    return output


def test_reach_dataset_triad_marker_orientation_sensitive(tmp_path):
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q2 = _quat_axis_angle([0, 1, 0], np.pi / 2)
    path = _build_zarr_with_goal_quat(tmp_path, [q1, q2])

    triad = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=path,
            horizon=2,
            n_obs_steps=2,
            goal_marker_points=8,
            goal_marker_radius=0.05,
            goal_marker_style="triad",
        ),
        split="all",
    )
    # Episode 0 (q1) vs episode 1 (q2): trailing goal slots must differ.
    m0 = triad[0]["obs"]["point_cloud"].numpy()[..., -8:, :]
    m_last = triad[len(triad) - 1]["obs"]["point_cloud"].numpy()[..., -8:, :]
    assert not np.allclose(m0, m_last, atol=1e-4)


def test_reach_dataset_sphere_marker_orientation_insensitive(tmp_path):
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q2 = _quat_axis_angle([0, 1, 0], np.pi / 2)
    path = _build_zarr_with_goal_quat(tmp_path, [q1, q2])

    sphere = ReachSequenceDataset(
        ReachDatasetConfig(
            dataset_path=path,
            horizon=2,
            n_obs_steps=2,
            goal_marker_points=8,
            goal_marker_radius=0.05,
            goal_marker_style="sphere",
        ),
        split="all",
    )
    m0 = sphere[0]["obs"]["point_cloud"].numpy()[..., -8:, :]
    m_last = sphere[len(sphere) - 1]["obs"]["point_cloud"].numpy()[..., -8:, :]
    # Same target position => identical sphere marker regardless of orientation.
    np.testing.assert_allclose(m0, m_last, atol=1e-6)
