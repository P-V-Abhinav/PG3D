"""Tests for in-place grasp roll + start-pose down-cone in the pose-reach writer.

Numeric assertions verify:
  * goal approach axes stay within the goal cone,
  * in-place roll psi spans ~[0, 180deg) when GRASP_ROLL_RANGE_DEG=180,
  * start orientations stay within the 10deg start down-cone,
  * sampling is deterministic under a seeded rng.

A guarded test also writes a 3D Plotly distribution diagram of the sampled
``link_tcp`` triads to ``artifacts/pose-reach-smoke/triad_distribution.html`` --
open it in a browser to visually confirm the triad frames are built correctly and
that approach directions fill the cone while the roll varies.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pg3d.policies.dp3.goal_markers import _quat_to_rotation_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_maniskill_pose_reach_dataset.py"


def _load_writer_module():
    spec = importlib.util.spec_from_file_location("_pose_reach_writer_roll", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"writer module import failed (missing deps): {exc}")
    return module


def _frame_axes(quat):
    """Return (x_axis, y_axis, z_axis) columns of R for a SAPIEN [w,x,y,z] quat."""
    R = _quat_to_rotation_matrix(np.asarray(quat, dtype=np.float64))
    return R[:, 0], R[:, 1], R[:, 2]


def _approach_axis(quat):
    # link_tcp +z (local approach) expressed in world = R @ [0,0,1] = 3rd column.
    return _frame_axes(quat)[2]


def _tilt_deg(quat):
    down = np.array([0.0, 0.0, -1.0])
    a = _approach_axis(quat)
    a = a / np.linalg.norm(a)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, down), -1.0, 1.0))))


def _roll_deg(quat, phi_ref_axis):
    """Signed roll of the frame x-axis about the approach axis vs a reference.

    ``phi_ref_axis`` is the canonical x_ref = [-sin(phi), cos(phi), 0] for the
    frame's azimuth; the roll is the angle from x_ref to the frame x-axis measured
    in the plane perpendicular to the approach axis.
    """
    x_axis, _y, z_axis = _frame_axes(quat)
    z_axis = z_axis / np.linalg.norm(z_axis)
    # Project reference and actual x into the plane perpendicular to z.
    def proj(v):
        v = v - np.dot(v, z_axis) * z_axis
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    ref = proj(phi_ref_axis)
    xa = proj(x_axis)
    cos_a = np.clip(np.dot(ref, xa), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_a))
    return float(ang)


def test_goal_orientations_within_cone_and_valid():
    module = _load_writer_module()
    rng = np.random.default_rng(0)
    half = 60.0
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=half, count=400)
    assert "downward" in oris
    max_tilt = 0.0
    for q in oris.values():
        q = np.asarray(q, dtype=np.float64)
        assert q.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5)
        max_tilt = max(max_tilt, _tilt_deg(q))
    assert max_tilt <= half + 1e-3


def test_downward_reference_is_straight_down_zero_roll():
    module = _load_writer_module()
    rng = np.random.default_rng(0)
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=60.0, count=5)
    assert _tilt_deg(oris["downward"]) < 1e-3


def test_in_place_roll_spans_full_range():
    module = _load_writer_module()
    # GRASP_ROLL_RANGE_DEG defaults to 180 in the module.
    assert module.GRASP_ROLL_RANGE_DEG == pytest.approx(180.0)
    rng = np.random.default_rng(1)
    half = 60.0
    n = 3000
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=half, count=n)
    rolls = []
    for name, q in oris.items():
        if name == "downward":
            continue
        # Recover the frame's azimuth from its approach axis to build x_ref.
        a = _approach_axis(q)
        phi = np.arctan2(a[1], a[0])
        x_ref = np.array([-np.sin(phi), np.cos(phi), 0.0])
        rolls.append(_roll_deg(q, x_ref))
    rolls = np.asarray(rolls)
    # Roll (unsigned, in [0,180]) should span nearly the whole range.
    assert rolls.min() < 15.0, f"roll min too high: {rolls.min():.1f}"
    assert rolls.max() > 165.0, f"roll max too low: {rolls.max():.1f}"
    # Reasonable spread across the range (not clustered).
    assert 60.0 < float(rolls.mean()) < 120.0, f"roll mean off-center: {rolls.mean():.1f}"


def test_roll_disabled_when_range_zero(monkeypatch):
    module = _load_writer_module()
    monkeypatch.setattr(module, "GRASP_ROLL_RANGE_DEG", 0.0)
    rng = np.random.default_rng(2)
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=60.0, count=200)
    # With roll disabled, every frame x-axis aligns with its x_ref (roll ~ 0).
    for name, q in oris.items():
        if name == "downward":
            continue
        a = _approach_axis(q)
        phi = np.arctan2(a[1], a[0])
        x_ref = np.array([-np.sin(phi), np.cos(phi), 0.0])
        assert _roll_deg(q, x_ref) < 1.0


def test_start_orientation_within_start_cone():
    module = _load_writer_module()
    rng = np.random.default_rng(3)
    half = module.START_APPROACH_CONE_HALF_ANGLE_DEG
    # Tunable global (see START_APPROACH_CONE_HALF_ANGLE_DEG); assert it is a
    # sane down-cone and that all sampled starts stay within it, rather than
    # pinning a specific degree value.
    assert 0.0 < half <= 90.0
    max_tilt = 0.0
    for _ in range(500):
        q = module._sample_start_orientation_in_cone(rng)
        q = np.asarray(q, dtype=np.float64)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5)
        max_tilt = max(max_tilt, _tilt_deg(q))
    assert max_tilt <= half + 1e-3, f"start tilt {max_tilt:.2f} exceeds {half}deg"


def test_sampler_deterministic():
    module = _load_writer_module()
    a = module.sample_equal_area_cone_orientations(
        np.random.default_rng(7), half_angle_deg=60.0, count=20
    )
    b = module.sample_equal_area_cone_orientations(
        np.random.default_rng(7), half_angle_deg=60.0, count=20
    )
    for k in a:
        np.testing.assert_allclose(a[k], b[k])


def test_write_triad_distribution_plotly():
    """Write a 3D Plotly diagram of sampled link_tcp triads (visual proof)."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    module = _load_writer_module()
    rng = np.random.default_rng(42)
    oris = module.sample_equal_area_cone_orientations(rng, half_angle_deg=60.0, count=150)

    # Scatter triad origins on the unit cap (approach direction) so the cone
    # coverage is visible; draw each frame's x/y/z axes as short colored segments.
    xs, ys, zs = [], [], []  # x-axis segments (red)
    ys_x, ys_y, ys_z = [], [], []  # y-axis segments (green)
    zs_x, zs_y, zs_z = [], [], []  # z-axis / approach segments (blue)
    approach_pts = []
    L = 0.12  # axis segment length in the plot
    for q in oris.values():
        x_axis, y_axis, z_axis = _frame_axes(q)
        origin = _approach_axis(q)  # place each triad at its approach direction on the cap
        approach_pts.append(origin)
        for seglist_x, seglist_y, seglist_z, axis in (
            (xs, ys, zs, x_axis),
            (ys_x, ys_y, ys_z, y_axis),
            (zs_x, zs_y, zs_z, z_axis),
        ):
            tip = origin + L * axis
            seglist_x.extend([origin[0], tip[0], None])
            seglist_y.extend([origin[1], tip[1], None])
            seglist_z.extend([origin[2], tip[2], None])

    ap = np.asarray(approach_pts)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", line=dict(color="red", width=3), name="x_EEF"
    ))
    fig.add_trace(go.Scatter3d(
        x=ys_x, y=ys_y, z=ys_z, mode="lines", line=dict(color="green", width=3), name="y_EEF"
    ))
    fig.add_trace(go.Scatter3d(
        x=zs_x, y=zs_y, z=zs_z, mode="lines", line=dict(color="blue", width=3),
        name="z_EEF (approach)",
    ))
    fig.add_trace(go.Scatter3d(
        x=ap[:, 0], y=ap[:, 1], z=ap[:, 2], mode="markers",
        marker=dict(size=2, color="gray"), name="approach dirs",
    ))
    fig.update_layout(
        title=(
            "Sampled link_tcp triads: approach dirs fill the 60deg down-cone, "
            "roll varies over [0,180). x=red y=green z=blue(approach)"
        ),
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        height=750, margin=dict(l=0, r=0, t=40, b=0),
    )

    out_dir = REPO_ROOT / "artifacts" / "pose-reach-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triad_distribution.html"
    fig.write_html(str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0
    print(f"wrote triad distribution diagram: {out_path}")
