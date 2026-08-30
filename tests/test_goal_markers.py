"""Tests for goal marker generation (sphere legacy + oriented triad)."""

from __future__ import annotations

import numpy as np
import pytest

from pg3d.policies.dp3.goal_markers import (
    DEFAULT_GOAL_MARKER_POINTS,
    DEFAULT_GOAL_MARKER_RADIUS,
    build_goal_marker,
    goal_marker_points,
    insert_goal_marker_points,
    triad_goal_marker,
    triad_goal_marker_offsets,
)

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # SAPIEN [w,x,y,z]


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a SAPIEN [w,x,y,z] quaternion for a rotation about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)


def test_triad_count_matches_default_budget() -> None:
    offsets = triad_goal_marker_offsets(IDENTITY_QUAT, num_points=DEFAULT_GOAL_MARKER_POINTS)
    assert offsets.shape == (DEFAULT_GOAL_MARKER_POINTS, 3)


@pytest.mark.parametrize("num_points", [4, 8, 32, 64, 192, 200])
def test_triad_count_exact_various_budgets(num_points: int) -> None:
    offsets = triad_goal_marker_offsets(IDENTITY_QUAT, num_points=num_points)
    assert offsets.shape == (num_points, 3)


def test_triad_identity_produces_axis_aligned_frame() -> None:
    offsets = triad_goal_marker_offsets(
        IDENTITY_QUAT, num_points=DEFAULT_GOAL_MARKER_POINTS, radius=1.0
    )
    # The farthest point should lie along +X (the longest arm) for identity.
    tip = offsets[np.argmax(np.linalg.norm(offsets, axis=1))]
    assert tip[0] > abs(tip[1]) and tip[0] > abs(tip[2])
    assert tip[0] > 0.0


def test_triad_has_no_rotational_symmetry() -> None:
    base = triad_goal_marker_offsets(IDENTITY_QUAT, num_points=192, radius=0.1)
    # A 180deg roll about each principal axis must change the ordered point set.
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        q = _quat_from_axis_angle(axis, np.pi)
        rotated = triad_goal_marker_offsets(q, num_points=192, radius=0.1)
        assert not np.allclose(base, rotated, atol=1e-4), (
            f"triad is symmetric under 180deg roll about {axis}"
        )
    # A small arbitrary rotation must also change the point set.
    q_small = _quat_from_axis_angle(np.array([0.3, 0.5, 0.8]), 0.2)
    assert not np.allclose(
        base, triad_goal_marker_offsets(q_small, num_points=192, radius=0.1), atol=1e-4
    )


def test_triad_rigidly_follows_position_and_orientation() -> None:
    target = np.array([0.3, -0.1, 0.4], dtype=np.float32)
    q = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 3)
    marker = triad_goal_marker(target, q, num_points=64, radius=0.05)
    assert marker.shape == (64, 3)
    # Marker centroid-of-arms should sit near the target (offsets are centered at 0).
    offsets = triad_goal_marker_offsets(q, num_points=64, radius=0.05)
    np.testing.assert_allclose(marker, target[None, :] + offsets, atol=1e-5)

    # Translating the target translates the whole marker rigidly.
    shift = np.array([0.1, 0.2, -0.05], dtype=np.float32)
    marker_shifted = triad_goal_marker(target + shift, q, num_points=64, radius=0.05)
    np.testing.assert_allclose(
        marker_shifted - marker, np.broadcast_to(shift, marker.shape), atol=1e-5
    )


def test_triad_is_rotation_of_canonical() -> None:
    # The rotated offsets should preserve pairwise distances (rigid rotation).
    base = triad_goal_marker_offsets(IDENTITY_QUAT, num_points=64, radius=0.1)
    q = _quat_from_axis_angle(np.array([0.2, 0.9, 0.1]), 1.1)
    rotated = triad_goal_marker_offsets(q, num_points=64, radius=0.1)
    base_norms = np.linalg.norm(base, axis=1)
    rotated_norms = np.linalg.norm(rotated, axis=1)
    np.testing.assert_allclose(base_norms, rotated_norms, atol=1e-5)


def test_build_goal_marker_dispatch() -> None:
    target = np.array([0.2, 0.2, 0.3], dtype=np.float32)
    sphere = build_goal_marker(target, style="sphere", num_points=64)
    sphere_ref = goal_marker_points(target, num_points=64)
    np.testing.assert_allclose(sphere, sphere_ref)

    with pytest.raises(ValueError):
        build_goal_marker(target, None, style="triad", num_points=64)

    with pytest.raises(ValueError):
        build_goal_marker(target, IDENTITY_QUAT, style="bogus", num_points=64)


def test_insert_goal_marker_triad_vs_sphere_orientation_sensitivity() -> None:
    rng = np.random.default_rng(0)
    pc = rng.standard_normal((256, 3)).astype(np.float32)
    target = np.array([0.3, 0.0, 0.4], dtype=np.float32)
    q1 = IDENTITY_QUAT
    q2 = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 2)

    # Sphere: trailing slots independent of orientation.
    s1 = insert_goal_marker_points(pc, target, num_points=64, style="sphere")
    s2 = insert_goal_marker_points(pc, target, num_points=64, style="sphere", quat=q2)
    np.testing.assert_allclose(s1[-64:], s2[-64:])

    # Triad: trailing slots change with orientation.
    t1 = insert_goal_marker_points(pc, target, num_points=64, style="triad", quat=q1)
    t2 = insert_goal_marker_points(pc, target, num_points=64, style="triad", quat=q2)
    assert not np.allclose(t1[-64:], t2[-64:], atol=1e-4)
    # Scene slots (non-goal) are untouched.
    np.testing.assert_allclose(t1[:-64], pc[:-64])


def test_triad_zero_points_and_zero_radius() -> None:
    assert triad_goal_marker_offsets(IDENTITY_QUAT, num_points=0).shape == (0, 3)
    zero_r = triad_goal_marker_offsets(IDENTITY_QUAT, num_points=16, radius=0.0)
    assert zero_r.shape == (16, 3)
    np.testing.assert_allclose(zero_r, 0.0)


def test_default_radius_constant_exposed() -> None:
    assert DEFAULT_GOAL_MARKER_RADIUS > 0
