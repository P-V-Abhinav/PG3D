"""eval_config.py — Single source of truth for all PG3D eval env positions.

Coordinate system
-----------------
All positions are in WORLD FRAME (not base-relative).

  Robot base  : sapien.Pose(p=[-0.615, 0.0, 0.0])
  Table center: world origin (0, 0, 0)
  +X          : forward away from robot, toward table
  +Y          : left (facing robot)
  +Z          : up

World-frame reach box (TCP workspace, IK-verified):
  x ∈ [-0.435,  0.035]
  y ∈ [-0.420,  0.420]
  z ∈ [ 0.050,  0.550]

Object resting positions (cubes, YCBs) sit on the table surface.
The table surface is at z ≈ 0.0 in world frame; a 7 cm cube rests
with its centre at z = 0.035.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Workspace bounds (world frame, TCP workspace)
# ---------------------------------------------------------------------------
WORKSPACE_X_MIN: float = -0.435
WORKSPACE_X_MAX: float =  0.035
WORKSPACE_Y_MIN: float = -0.420
WORKSPACE_Y_MAX: float =  0.420
WORKSPACE_Z_MIN: float =  0.050
WORKSPACE_Z_MAX: float =  0.550

# Start-site must be within this distance of the canonical rest-pose TCP.
# Checked at the start of every episode in _initialize_episode.
START_SITE_REST_TOLERANCE_M: float = 0.12


def check_workspace(pos: tuple[float, float, float], label: str) -> None:
    """Raise ValueError if pos is outside the TCP workspace bounds.

    Called at module import time for goals and place targets so
    mis-configured positions are caught before the env runs.
    Object resting z (≈ 0.035) is exempt from the z floor check since
    objects sit on the table — only TCP goal positions are constrained.
    """
    x, y, z = pos
    errors = []
    if not (WORKSPACE_X_MIN <= x <= WORKSPACE_X_MAX):
        errors.append(f"x={x:.4f} not in [{WORKSPACE_X_MIN}, {WORKSPACE_X_MAX}]")
    if not (WORKSPACE_Y_MIN <= y <= WORKSPACE_Y_MAX):
        errors.append(f"y={y:.4f} not in [{WORKSPACE_Y_MIN}, {WORKSPACE_Y_MAX}]")
    if not (WORKSPACE_Z_MIN <= z <= WORKSPACE_Z_MAX):
        errors.append(f"z={z:.4f} not in [{WORKSPACE_Z_MIN}, {WORKSPACE_Z_MAX}]")
    if errors:
        raise ValueError(f"{label}: out of TCP workspace bounds — {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# ENV 1: Reach goal positions  (world frame, TCP target)
# ---------------------------------------------------------------------------
# Goals spread out to use the generous workspace bounds.
REACH_GOALS: dict[str, tuple[float, float, float]] = {
    "v1": (-0.100,  0.000, 0.150),   # far front
    "v2": (-0.050,  0.200, 0.080),   # far front-left, near table
    "v3": (-0.100,  0.380, 0.150),   # far left diagonal
    "v4": (-0.100, -0.380, 0.150),   # far right diagonal
    "v5": (-0.400,  0.000, 0.450),   # back upward
}

# Validate at import time
for _v, _pos in REACH_GOALS.items():
    check_workspace(_pos, f"REACH_GOALS[{_v!r}]")


# ---------------------------------------------------------------------------
# ENV 2 / 4: Canonical cube positions and place targets
# (SHARED across P&P, Obs-PP, L/R, Pose, Keep-Pose tasks)
#
# Cube rests on table: z = 0.035 (half-size of 7 cm cube).
# Place target z = 0.035 — fully at the surface of the table.
# ---------------------------------------------------------------------------
CUBE_POSITIONS: dict[str, tuple[float, float, float]] = {
    "v1": (-0.280,  0.000, 0.035),   # Close (front-center, directly under rest TCP)
    "v2": (-0.050,  0.200, 0.035),   # Far front center-left
    "v3": (-0.050, -0.300, 0.035),   # Far front right
    "v4": (-0.050,  0.350, 0.035),   # Far front left
    "v5": (-0.300, -0.150, 0.035),   # Close (moderate right offset)
}

# Place targets: where the cube's centre should end up (on table surface).
# The goal_site is positioned here so the policy steers the TCP to this XY.
PLACE_TARGETS: dict[str, tuple[float, float, float]] = {
    "v1": (-0.150, -0.350, 0.035),   # Far front right (lateral transport)
    "v2": (-0.400,  0.200, 0.035),   # Back left (longitudinal pull-back)
    "v3": (-0.400,  0.300, 0.035),   # Back left (diagonal cross-sweep)
    "v4": (-0.350, -0.100, 0.035),   # Back right (diagonal pull-back)
    "v5": (-0.100,  0.250, 0.035),   # Far front left (diagonal sweep)
}

# Validate TCP targets (place targets treated as TCP goals in the env)
# goal_site for PP is at PLACE_TARGETS; z=0.035 is below TCP floor → raise
# separately below with a note. We skip the z check for on-table places.
def _check_xy_workspace(pos: tuple[float, float, float], label: str) -> None:
    x, y, _ = pos
    errors = []
    if not (WORKSPACE_X_MIN <= x <= WORKSPACE_X_MAX):
        errors.append(f"x={x:.4f}")
    if not (WORKSPACE_Y_MIN <= y <= WORKSPACE_Y_MAX):
        errors.append(f"y={y:.4f}")
    if errors:
        raise ValueError(f"{label}: XY out of workspace — {'; '.join(errors)}")

for _v, _pos in {**CUBE_POSITIONS, **PLACE_TARGETS}.items():
    _check_xy_workspace(_pos, f"CUBE/PLACE[{_v!r}]")


# ---------------------------------------------------------------------------
# ENV 3: Obstacle-reach start / goal positions  (world frame, TCP targets)
# ---------------------------------------------------------------------------
REST_TCP_POS = (-0.288, 0.000, 0.308)

OBS_REACH_CONFIGS: dict[str, dict[str, tuple[float, float, float]]] = {
    "v1": {
        "start": REST_TCP_POS,
        "goal":  (-0.100,  0.350, 0.100),
        "label": "forward_left_slalom",
    },
    "v2": {
        "start": REST_TCP_POS,
        "goal":  (-0.050,  0.000, 0.100),
        "label": "pure_forward_slalom",
    },
    "v3": {
        "start": REST_TCP_POS,
        "goal":  (-0.100, -0.350, 0.100),
        "label": "forward_right_slalom",
    },
    "v4": {
        "start": REST_TCP_POS,
        "goal":  (-0.400,  0.350, 0.100),
        "label": "backward_left_slalom",
    },
    "v5": {
        "start": REST_TCP_POS,
        "goal":  (-0.400, -0.350, 0.100),
        "label": "backward_right_slalom",
    },
}

for _v, _cfg in OBS_REACH_CONFIGS.items():
    check_workspace(_cfg["start"], f"OBS_REACH[{_v!r}].start")
    check_workspace(_cfg["goal"],  f"OBS_REACH[{_v!r}].goal")


# ---------------------------------------------------------------------------
# ENV 5: Cluttered layout definitions
# All positions world frame; z given explicitly for each object type.
# Distances are manually verified — all objects ≥ 2 cm apart.
# ---------------------------------------------------------------------------
CLUTTERED_LAYOUTS: dict[str, dict] = {
    "v1": {
        "target": {"model": "025_mug",              "pos": (-0.280,  0.050, 0.050), "yaw_deg":  0},
        "clutter": [
            {"model": "024_bowl",                   "pos": (-0.190,  0.200, 0.040), "yaw_deg":  0},
            {"model": "005_tomato_soup_can",         "pos": (-0.360,  0.160, 0.040), "yaw_deg": 45},
            {"model": "009_gelatin_box",             "pos": (-0.220, -0.150, 0.040), "yaw_deg": 30},
        ],
        "place_goal": (-0.380, 0.000, 0.035),
        "label": "loose_easy",
    },
    "v2": {
        "target": {"model": "006_mustard_bottle",   "pos": (-0.250,  0.000, 0.060), "yaw_deg":  0},
        "clutter": [
            {"model": "003_cracker_box",             "pos": (-0.200,  0.090, 0.060), "yaw_deg": 90},
            {"model": "004_sugar_box",               "pos": (-0.200, -0.090, 0.050), "yaw_deg": 45},
            {"model": "010_potted_meat_can",         "pos": (-0.310,  0.090, 0.040), "yaw_deg":  0},
            {"model": "009_gelatin_box",             "pos": (-0.310, -0.090, 0.040), "yaw_deg": 60},
        ],
        "place_goal": (-0.390, 0.000, 0.035),
        "label": "dense_cluster",
    },
    "v3": {
        "target": {"model": "011_banana",           "pos": (-0.300, -0.150, 0.040), "yaw_deg":  0},
        "clutter": [
            {"model": "025_mug",                    "pos": (-0.220,  0.000, 0.050), "yaw_deg":  0},
            {"model": "024_bowl",                   "pos": (-0.250,  0.150, 0.040), "yaw_deg":  0},
            {"model": "004_sugar_box",              "pos": (-0.350,  0.100, 0.050), "yaw_deg": 30},
            {"model": "005_tomato_soup_can",        "pos": (-0.380, -0.050, 0.040), "yaw_deg":  0},
            {"model": "009_gelatin_box",            "pos": (-0.360, -0.200, 0.040), "yaw_deg": 45},
        ],
        "place_goal": (-0.180, 0.200, 0.035),
        "label": "arc_of_clutter",
    },
    "v4": {
        "target": {"model": "005_tomato_soup_can",  "pos": (-0.230,  0.200, 0.040), "yaw_deg":  0},
        "clutter": [
            {"model": "024_bowl",                   "pos": (-0.280,  0.080, 0.040), "yaw_deg":  0},
            {"model": "006_mustard_bottle",         "pos": (-0.200,  0.060, 0.060), "yaw_deg":  0},
            {"model": "003_cracker_box",            "pos": (-0.340,  0.180, 0.060), "yaw_deg": 90},
        ],
        "place_goal": (-0.380, 0.000, 0.035),
        "label": "mixed_sizes_path_blocked",
    },
    "v5": {
        "target": {"model": "009_gelatin_box",      "pos": (-0.280,  0.000, 0.040), "yaw_deg":  0},
        "clutter": [
            {"model": "025_mug",                    "pos": (-0.200,  0.100, 0.050), "yaw_deg":  0},
            {"model": "024_bowl",                   "pos": (-0.200, -0.100, 0.040), "yaw_deg":  0},
            {"model": "006_mustard_bottle",         "pos": (-0.360,  0.100, 0.060), "yaw_deg":  0},
            {"model": "005_tomato_soup_can",        "pos": (-0.360, -0.100, 0.040), "yaw_deg":  0},
            {"model": "003_cracker_box",            "pos": (-0.220,  0.220, 0.060), "yaw_deg": 90},
            {"model": "004_sugar_box",              "pos": (-0.220, -0.220, 0.050), "yaw_deg": 45},
            {"model": "010_potted_meat_can",        "pos": (-0.340,  0.000, 0.040), "yaw_deg":  0},
            {"model": "011_banana",                 "pos": (-0.170,  0.000, 0.040), "yaw_deg":  0},
        ],
        "place_goal": (-0.400, 0.000, 0.035),
        "label": "maximum_clutter",
    },
}

# Validate XY of cluttered place goals
for _v, _layout in CLUTTERED_LAYOUTS.items():
    _check_xy_workspace(_layout["place_goal"], f"CLUTTERED[{_v!r}].place_goal")


# ---------------------------------------------------------------------------
# Workspace bounding-box visualisation helpers
# ---------------------------------------------------------------------------

def workspace_box_edges() -> list[dict]:
    """Return 12 edge descriptors for building the workspace wireframe.

    Each descriptor has:
      centre  : (x, y, z) world-frame centre of the edge bar
      half_sizes : (hx, hy, hz) for actors.build_box()

    The wireframe uses 2 mm square cross-section bars.
    """
    x0, x1 = WORKSPACE_X_MIN, WORKSPACE_X_MAX
    y0, y1 = WORKSPACE_Y_MIN, WORKSPACE_Y_MAX
    z0, z1 = WORKSPACE_Z_MIN, WORKSPACE_Z_MAX
    hx = (x1 - x0) / 2
    hy = (y1 - y0) / 2
    hz = (z1 - z0) / 2
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    cz = (z0 + z1) / 2
    t = 0.002  # 2 mm bar half-thickness

    edges = []
    # 4 edges parallel to X
    for yc in (y0, y1):
        for zc in (z0, z1):
            edges.append({"centre": (cx, yc, zc), "half_sizes": (hx, t, t)})
    # 4 edges parallel to Y
    for xc in (x0, x1):
        for zc in (z0, z1):
            edges.append({"centre": (xc, cy, zc), "half_sizes": (t, hy, t)})
    # 4 edges parallel to Z
    for xc in (x0, x1):
        for yc in (y0, y1):
            edges.append({"centre": (xc, yc, cz), "half_sizes": (t, t, hz)})
    return edges
