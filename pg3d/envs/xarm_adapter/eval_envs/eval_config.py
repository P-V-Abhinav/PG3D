"""eval_config.py -- Single source of truth for every PG3D eval env's geometry.

Coordinate system
-----------------
All positions here are WORLD frame, and in the active (M2) layout the world
frame *is* the xArm7 base frame: the robot is bolted at ``(0, 0, 0)`` and
``reach_env.TABLE_ORIGIN_SHIFT_X`` moves ManiSkill's table so the
robot-to-table arrangement matches the layout the reach datasets were
generated under (ADR 0014 / ADR 0022).

  Robot base  : (0, 0, 0)
  +X          : forward, away from the robot, across the table
  +Y          : robot's left
  +Z          : up

Provenance of the numbers
-------------------------
The suite was authored against the OLD (M1) simulator layout, where the base
sat at ``(-0.615, 0, 0)`` and the table was at ManiSkill's own default pose.
Every literal below is kept verbatim as the ``_M1_*`` value it was frozen at
and converted once, here, by adding the repo seam's ``FRAME_SHIFT``
(= ``+0.615`` in x; see ``master_env_settings``). Because the active layout
shifts the *table* by exactly the same amount, that single conversion preserves
both the robot-relative and the table-relative geometry of every goal, cube,
obstacle and clutter object -- the scene is unchanged, only the frame it is expressed in moved.

Do NOT re-tune the ``_M1_*`` literals to "nicer" M2 numbers. They are the
frozen evaluation population; the shift is a relabeling of the same scene.

Reach-box caveat (read before adding positions)
-----------------------------------------------
``XARM7_REACH_BOX_BASE`` in reach_config is a deliberately conservative
*sampling* box (dx <= 0.50, dz <= 0.37) trimmed against a gripper
self-collision blind spot. This suite was laid out against the larger box the
training dataset actually sampled from (dx_hi 0.69, dy +/-0.45, dz_hi 0.53), so
11 of the frozen positions sit outside the trimmed box while staying inside
both the training box and the point-cloud crop box. That is intentional and is
reported by :func:`workspace_report`; positions are hard-validated against the
crop box only, since a position outside the crop is invisible to the policy.
"""

from __future__ import annotations

import warnings

import numpy as np

from .master_env_settings import CROP_BOUNDS as _REPO_CROP_BOUNDS
from .master_env_settings import FRAME_SHIFT as _REPO_FRAME_SHIFT
from .master_env_settings import REACH_BOUNDS as _REPO_REACH_BOUNDS
Vec3 = tuple[float, float, float]

# ---------------------------------------------------------------------------
# Frame conversion (M1 authoring frame -> active M2 world/base frame)
# ---------------------------------------------------------------------------
EVAL_FRAME_SHIFT: np.ndarray = np.asarray(_REPO_FRAME_SHIFT, dtype=np.float32)


def to_world(pos_m1: Vec3) -> Vec3:
    """Convert one frozen M1-frame position to the active world frame.

    Done in float64 and rounded to micrometres so the frozen numbers stay
    exact-looking in manifests instead of carrying float32 dust (0.327000021...).
    """
    shifted = np.asarray(pos_m1, dtype=np.float64) + EVAL_FRAME_SHIFT.astype(np.float64)
    return tuple(round(float(value), 6) for value in shifted)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Goal-marker contract (must match the checkpoint's dataset bake)
# ---------------------------------------------------------------------------
# The active checkpoint is trained on a SPHERE marker: 192 points placed on a
# shell of radius 0.055 m by a golden-angle spiral. The encoder flattens the
# marker slots instead of pooling them, so shape/count/radius are part of the
# policy's input contract, not cosmetics -- see goal_markers.goal_marker_offsets
# and ADR 0022. These re-export the package defaults so an env, an eval script
# and the dataset writer cannot drift apart silently.
# These are stated literally rather than imported from
# ``pg3d.policies.dp3.goal_markers`` so that importing this module never pulls
# in torch (the reason ``pg3d.tasks`` keeps its simulator imports function-local
# too). :func:`verify_marker_contract` closes the drift risk by checking them
# against the policy package's own defaults, and every eval env calls it at
# construction.
# Measured out of the checkpoint's own dataset (`pose_variety_final.zarr`) on
# 2026-09-09 by matching the trailing points of `data/point_cloud` against
# `goal_marker_offsets`: sphere / 192 / 0.045, at 0.000000 m error. Re-measure
# the same way if the dataset behind the checkpoint changes.
GOAL_MARKER_POINTS: int = 192
GOAL_MARKER_RADIUS: float = 0.045
GOAL_MARKER_SHAPE: str = "sphere"


def verify_marker_contract() -> None:
    """Raise if the policy package's marker defaults have moved away from ours.

    The encoder flattens the marker slots, so the marker's count, radius and
    shape are part of the checkpoint's input contract. A silent divergence here
    would feed the policy a goal cue it never trained on and nothing else would
    complain -- so this fails closed instead.
    """
    from pg3d.policies.dp3.goal_markers import (
        DEFAULT_GOAL_MARKER_POINTS,
        DEFAULT_GOAL_MARKER_RADIUS,
        DEFAULT_GOAL_MARKER_SHAPE,
    )

    mismatches = {
        name: (mine, theirs)
        for name, mine, theirs in (
            ("num_points", GOAL_MARKER_POINTS, int(DEFAULT_GOAL_MARKER_POINTS)),
            ("radius", GOAL_MARKER_RADIUS, float(DEFAULT_GOAL_MARKER_RADIUS)),
            ("shape", GOAL_MARKER_SHAPE, str(DEFAULT_GOAL_MARKER_SHAPE)),
        )
        if mine != theirs
    }
    if mismatches:
        raise RuntimeError(
            "goal-marker contract mismatch between eval_config and "
            f"pg3d.policies.dp3.goal_markers: {mismatches}. The eval envs and the "
            "checkpoint's dataset bake must agree on count/radius/shape."
        )

# ---------------------------------------------------------------------------
# Start poses
# ---------------------------------------------------------------------------
# The suite's default start TCP, frozen as an M1 number and converted like
# everything else. The manipulation families (T3-T10) all start here: their
# object layouts were arranged around it.
#
# It is NOT the "rest" keyframe's TCP. The rest keyframe
# (qpos [0, -0.4, 0, 0.5, 0, 0.9, 0]) puts link_tcp at (0.327, 0.000, 0.136);
# the frozen number is 0.172 m higher, which is exactly the gripper's TCP
# offset -- i.e. the M1 value was recorded against the NO-GRIPPER variant, whose
# TCP is the flange. Measured on 2026-09-08. Every start is therefore reached by
# a joint configuration, baked below.
_M1_START_TCP: Vec3 = (-0.288, 0.000, 0.308)
START_TCP: Vec3 = to_world(_M1_START_TCP)

# How far the achieved start TCP may sit from its declared value before the env
# raises rather than run a mis-posed episode.
START_TCP_TOLERANCE_M: float = 0.02

# Per-variant start TCPs. Every one of the 25 envs starts the arm somewhere
# different -- a same-start-everywhere suite cannot distinguish a
# policy that generalises over initial configurations from one that memorised
# one -- while staying close to the rest pose so no variant begins in a
# contorted or near-singular configuration.
#
# Chosen in the ACTIVE frame (not converted from M1, unlike the frozen scene
# geometry): each is a small offset from START_TCP and inside the IK-verified
# reach box. Verified 2026-09-08: all 25 IK-solve to 0.01 mm with |dq| from the
# rest keyframe of 0.41-0.94 rad; every obstacle-reach start clears every bar by
# >= 7.3 cm at the TCP; and a physics step at each start reports no contact
# between the robot and any obstacle, cube or clutter object.
REACH_START_TCPS: dict[str, Vec3] = {
    "v1": (0.327, 0.100, 0.308),   # rest height, 10 cm left
    "v2": (0.327, -0.100, 0.308),  # rest height, 10 cm right
    "v3": (0.267, 0.070, 0.258),   # back-left, lower
    "v4": (0.267, -0.070, 0.258),  # back-right, lower
    # v5 runs the workspace diagonal INWARD: it starts high on the front-left
    # side and reaches REACH_GOALS["v5"] on the opposite back-right corner of
    # the eval box in xy (x_min, y_min). World frame, unlike the goals below.
    #
    # NOT the box's front-left corner (0.650, 0.420, 0.500): that is 0.921 m
    # radially from the base and the planner's IK misses it by 0.219 m with the
    # frozen tool-down orientation -- it is off the arm's reachable set, not
    # merely awkward, so an env starting there raises at every reset. This is
    # the reachable stand-in on the same diagonal (verified 2026-09-09: IK
    # solves to 5 um, |dq| 2.67 rad from the rest keyframe).
    "v5": (0.480, 0.300, 0.420),   # front-left and high -- diagonal start
}
OBS_REACH_START_TCPS: dict[str, Vec3] = {
    "v1": (0.297, 0.080, 0.278),   # back-left of the slalom entry
    "v2": (0.297, -0.080, 0.278),  # back-right of the slalom entry
    "v3": (0.357, 0.060, 0.278),   # forward-left
    "v4": (0.357, -0.060, 0.278),  # forward-right
    "v5": (0.327, 0.000, 0.238),   # centred and low
}
PP_START_TCPS: dict[str, Vec3] = {
    "v1": (0.307, 0.060, 0.328),
    "v2": (0.307, -0.060, 0.328),
    "v3": (0.367, 0.040, 0.298),
    "v4": (0.367, -0.040, 0.298),
    "v5": (0.287, 0.000, 0.288),
}
OBS_PP_START_TCPS: dict[str, Vec3] = {
    "v1": (0.347, 0.070, 0.288),
    "v2": (0.347, -0.070, 0.288),
    "v3": (0.277, 0.050, 0.318),
    "v4": (0.277, -0.050, 0.318),
    # Moved from (0.327, 0.090, 0.258) on 2026-09-08: at that start the right
    # finger was in contact with the middle slalom bar (555 N in the first
    # physics step). This one clears every bar by >= 13 cm at the TCP.
    "v5": (0.297, -0.090, 0.288),
}
CLUTTERED_START_TCPS: dict[str, Vec3] = {
    "v1": (0.317, 0.045, 0.338),
    "v2": (0.317, -0.045, 0.338),
    "v3": (0.377, 0.020, 0.318),
    "v4": (0.257, 0.020, 0.288),
    "v5": (0.337, -0.090, 0.298),
}

# Arm joint configuration (joint1..joint7) that puts link_tcp at each start.
#
# Baked rather than solved at reset, for three reasons: a frozen suite should
# not depend on a planner being installed; mplib IK returns several branches and
# picking a different one between runs would silently change the arm's whole
# posture; and IK inside `_initialize_episode` costs seconds on every episode.
# Solved with mplib against the rest-pose downward tool orientation
# (quaternion [0, 1, 0, 0] wxyz), seeded from the rest keyframe, then wrapped to
# the branch closest to it in joint space. Gripper joints are NOT included --
# each env keeps its own jaw width.
#
# Regenerate these if the URDF, the TCP link or the base pose ever changes: the
# env checks the achieved TCP against the declared one at every reset and raises
# on a mismatch, so a stale bake fails loudly rather than quietly.
SUITE_START_QPOS: tuple[float, ...] = (
    -0.000006, -0.516128, -0.000004, 0.915525, -0.000002, 1.431653, 0.000002,
)
REACH_START_QPOS: dict[str, tuple[float, ...]] = {
    "v1": (0.092365, -0.471823, 0.158598, 0.954240, 0.072656, 1.421358, 0.223054),
    "v2": (-0.092376, -0.471822, -0.158605, 0.954242, -0.072659, 1.421359, -0.223049),
    "v3": (0.090691, -0.716274, 0.123908, 0.649485, 0.083033, 1.362648, 0.167171),
    "v4": (-0.090703, -0.716272, -0.123919, 0.649486, -0.083040, 1.362648, -0.167167),
    # Re-baked 2026-09-09 for v5's inverted diagonal start (0.480, 0.300,
    # 0.420): the least-contorted of the planner's 115 IK solutions, after
    # unwrapping each joint toward the rest keyframe within its limits
    # (|dq| 2.67 rad; FK reproduces the declared start TCP to 5 um).
    "v5": (0.526222, 0.595511, 0.085841, 2.579609, -0.052562, 1.985298, 0.576168),
}
OBS_REACH_START_QPOS: dict[str, tuple[float, ...]] = {
    "v1": (0.087553, -0.596797, 0.133724, 0.779618, 0.076496, 1.372819, 0.183306),
    "v2": (-0.087564, -0.596795, -0.133733, 0.779620, -0.076501, 1.372819, -0.183302),
    "v3": (0.054996, -0.399404, 0.092815, 0.921121, 0.037221, 1.319149, 0.131281),
    "v4": (-0.055007, -0.399403, -0.092820, 0.921122, -0.037225, 1.319149, -0.131276),
    "v5": (-0.000006, -0.511849, -0.000004, 0.718899, -0.000004, 1.230748, 0.000002),
}
PP_START_QPOS: dict[str, tuple[float, ...]] = {
    "v1": (0.057697, -0.560114, 0.096652, 0.944854, 0.051411, 1.502953, 0.136175),
    "v2": (-0.057708, -0.560113, -0.096660, 0.944855, -0.051414, 1.502953, -0.136170),
    "v3": (0.034953, -0.373851, 0.060695, 0.998091, 0.022601, 1.371365, 0.086997),
    "v4": (-0.034964, -0.373851, -0.060700, 0.998092, -0.022605, 1.371366, -0.086992),
    "v5": (-0.000006, -0.663648, -0.000005, 0.761126, -0.000003, 1.424774, 0.000002),
}
OBS_PP_START_QPOS: dict[str, tuple[float, ...]] = {
    "v1": (0.064599, -0.428639, 0.109237, 0.927037, 0.046418, 1.353647, 0.154024),
    "v2": (-0.064610, -0.428638, -0.109243, 0.927038, -0.046422, 1.353647, -0.154019),
    "v3": (0.054280, -0.673520, 0.083823, 0.843605, 0.052328, 1.515486, 0.116975),
    "v4": (-0.054291, -0.673519, -0.083833, 0.843606, -0.052332, 1.515486, -0.116970),
    "v5": (-0.095584, -0.587260, -0.149494, 0.815666, -0.083864, 1.398361, -0.205883),
}
CLUTTERED_START_QPOS: dict[str, tuple[float, ...]] = {
    "v1": (0.041446, -0.527986, 0.071450, 0.994681, 0.036018, 1.521588, 0.101431),
    "v2": (-0.041456, -0.527985, -0.071458, 0.994682, -0.036020, 1.521588, -0.101426),
    "v3": (0.016700, -0.341630, 0.029840, 1.081867, 0.010105, 1.423364, 0.043342),
    "v4": (0.025540, -0.772286, 0.035174, 0.700522, 0.024661, 1.472528, 0.048333),
    "v5": (-0.082969, -0.448186, -0.141595, 0.942259, -0.062243, 1.386878, -0.199331),
}


# ---------------------------------------------------------------------------
# Workspace bounds
# ---------------------------------------------------------------------------
# The eval box this suite was laid out inside, expressed in the active frame.
_M1_EVAL_BOX = np.array(
    [
        [-0.435, 0.035],   # x
        [-0.420, 0.420],   # y
        [0.050, 0.550],    # z
    ],
    dtype=np.float32,
)
EVAL_WORKSPACE_BOUNDS: np.ndarray = _M1_EVAL_BOX + np.asarray(
    [EVAL_FRAME_SHIFT[0], EVAL_FRAME_SHIFT[1], EVAL_FRAME_SHIFT[2]], dtype=np.float32
).reshape(3, 1)

WORKSPACE_X_MIN, WORKSPACE_X_MAX = (float(v) for v in EVAL_WORKSPACE_BOUNDS[0])
WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = (float(v) for v in EVAL_WORKSPACE_BOUNDS[1])
WORKSPACE_Z_MIN, WORKSPACE_Z_MAX = (float(v) for v in EVAL_WORKSPACE_BOUNDS[2])

#: Point-cloud crop box (world frame). A position outside this is not merely
#: hard to reach, it is invisible to the policy -- so this is the hard gate.
CROP_BOUNDS: np.ndarray = np.asarray(_REPO_CROP_BOUNDS, dtype=np.float32)

#: The conservative IK-verified sampling box. Advisory only here; see the module
#: docstring for why a number of frozen positions sit outside it (the exact
#: list is reported by :func:`workspace_report`).
REACH_BOUNDS: np.ndarray = np.asarray(_REPO_REACH_BOUNDS, dtype=np.float32)


def _outside(pos: Vec3, bounds: np.ndarray, *, axes: str = "xyz") -> list[str]:
    """Return per-axis messages for each requested axis that falls outside bounds."""
    box = np.asarray(bounds, dtype=np.float32).reshape(3, 2)
    errors: list[str] = []
    for index, axis in enumerate("xyz"):
        if axis not in axes:
            continue
        value = float(pos[index])
        low, high = float(box[index, 0]), float(box[index, 1])
        if not (low <= value <= high):
            errors.append(f"{axis}={value:.4f} not in [{low:.4f}, {high:.4f}]")
    return errors


def check_workspace(pos: Vec3, label: str, *, axes: str = "xyz") -> None:
    """Raise if ``pos`` is outside the crop box; warn if outside the reach box.

    ``axes`` lets on-table object positions skip the z check: an object rests at
    z ~= 0.02--0.06 by its own geometry, which is not a TCP-reachability claim.
    """
    hard = _outside(pos, CROP_BOUNDS, axes=axes)
    if hard:
        raise ValueError(
            f"{label}: outside the point-cloud crop box -- the policy cannot see it "
            f"({'; '.join(hard)})"
        )
    soft = _outside(pos, REACH_BOUNDS, axes=axes)
    if soft:
        _OUT_OF_REACH_BOX.append((label, tuple(float(v) for v in pos), tuple(soft)))


#: Every position that validated against the crop box but sits outside the
#: conservative IK-verified reach box. Populated at import; reported by
#: :func:`workspace_report` so a run can record the caveat instead of
#: rediscovering it.
_OUT_OF_REACH_BOX: list[tuple[str, tuple[float, ...], tuple[str, ...]]] = []


def workspace_report() -> dict[str, object]:
    """Return the import-time bounds audit, for logging into a run manifest."""
    return {
        "frame": "xarm7_base == world (M2)",
        "frame_shift_from_authoring_frame": EVAL_FRAME_SHIFT.tolist(),
        "eval_box": EVAL_WORKSPACE_BOUNDS.tolist(),
        "crop_box": CROP_BOUNDS.tolist(),
        "verified_reach_box": REACH_BOUNDS.tolist(),
        "outside_verified_reach_box": [
            {"label": label, "position": list(position), "axes": list(axes)}
            for label, position, axes in _OUT_OF_REACH_BOX
        ],
    }


# Start poses are validated here, once check_workspace exists.
for _v, _pos in REACH_START_TCPS.items():
    check_workspace(_pos, f"REACH_START_TCPS[{_v!r}]")
for _name, _table in (
    ("OBS_REACH_START_TCPS", OBS_REACH_START_TCPS),
    ("PP_START_TCPS", PP_START_TCPS),
    ("OBS_PP_START_TCPS", OBS_PP_START_TCPS),
    ("CLUTTERED_START_TCPS", CLUTTERED_START_TCPS),
):
    for _v, _pos in _table.items():
        check_workspace(_pos, f"{_name}[{_v!r}]")
check_workspace(START_TCP, "START_TCP")


# ---------------------------------------------------------------------------
# T1 -- Reach goal positions (TCP targets)
# ---------------------------------------------------------------------------
_M1_REACH_GOALS: dict[str, Vec3] = {
    "v1": (-0.100, 0.000, 0.150),   # far front
    "v2": (-0.050, 0.200, 0.080),   # far front-left, near the table
    "v3": (-0.100, 0.380, 0.150),   # far left diagonal
    "v4": (-0.100, -0.380, 0.150),  # far right diagonal
    # M1 FRAME -- to_world adds +0.615 to x, so this lands on the eval box's
    # back-right corner at world (0.180, -0.420, 0.348), the opposite xy corner
    # from REACH_START_TCPS["v5"]. On the boundary of the conservative
    # IK-verified reach box (the import-time audit reports it) and inside the
    # crop box, so the policy can see the goal marker.
    "v5": (-0.435, -0.420, 0.348),  # back-right workspace corner
}
REACH_GOALS: dict[str, Vec3] = {k: to_world(v) for k, v in _M1_REACH_GOALS.items()}

for _v, _pos in REACH_GOALS.items():
    check_workspace(_pos, f"REACH_GOALS[{_v!r}]")


# ---------------------------------------------------------------------------
# T3 / T4 / T5 / T6 / T7 / T8 -- cube positions and place targets
#
# The cube is a 7 cm box resting on the table, so its centre is at z = 0.035.
# Place targets are where the cube's centre should end up.
# ---------------------------------------------------------------------------
CUBE_HALF_SIZE: float = 0.035

_M1_CUBE_POSITIONS: dict[str, Vec3] = {
    "v1": (-0.280, 0.000, 0.035),   # close, front-centre, under the rest TCP
    "v2": (-0.050, 0.200, 0.035),   # far front, centre-left
    "v3": (-0.050, -0.300, 0.035),  # far front right
    "v4": (-0.050, 0.350, 0.035),   # far front left
    "v5": (-0.300, -0.150, 0.035),  # close, moderate right offset
}
_M1_PLACE_TARGETS: dict[str, Vec3] = {
    "v1": (-0.150, -0.350, 0.035),  # far front right (lateral transport)
    "v2": (-0.400, 0.200, 0.035),   # back left (longitudinal pull-back)
    "v3": (-0.400, 0.300, 0.035),   # back left (diagonal cross-sweep)
    "v4": (-0.350, -0.100, 0.035),  # back right (diagonal pull-back)
    "v5": (-0.100, 0.250, 0.035),   # far front left (diagonal sweep)
}
CUBE_POSITIONS: dict[str, Vec3] = {k: to_world(v) for k, v in _M1_CUBE_POSITIONS.items()}
PLACE_TARGETS: dict[str, Vec3] = {k: to_world(v) for k, v in _M1_PLACE_TARGETS.items()}

for _v, _pos in CUBE_POSITIONS.items():
    check_workspace(_pos, f"CUBE_POSITIONS[{_v!r}]", axes="xy")
for _v, _pos in PLACE_TARGETS.items():
    check_workspace(_pos, f"PLACE_TARGETS[{_v!r}]", axes="xy")


# ---------------------------------------------------------------------------
# T2 -- Obstacle-reach start / goal pairs (TCP targets)
#
# `start` is this variant's own start (OBS_REACH_START_TCPS). The three bars are
# placed on the start->goal path, so the start and the obstacle layout move
# together and stay mutually clear.
# ---------------------------------------------------------------------------
_M1_OBS_REACH_GOALS: dict[str, tuple[Vec3, str]] = {
    "v1": ((-0.100, 0.350, 0.100), "forward_left_slalom"),
    "v2": ((-0.050, 0.000, 0.100), "pure_forward_slalom"),
    "v3": ((-0.100, -0.350, 0.100), "forward_right_slalom"),
    "v4": ((-0.400, 0.350, 0.100), "backward_left_slalom"),
    "v5": ((-0.400, -0.350, 0.100), "backward_right_slalom"),
}
OBS_REACH_CONFIGS: dict[str, dict[str, object]] = {
    key: {
        "start": OBS_REACH_START_TCPS[key],
        "goal": to_world(goal),
        "label": label,
    }
    for key, (goal, label) in _M1_OBS_REACH_GOALS.items()
}

for _v, _cfg in OBS_REACH_CONFIGS.items():
    check_workspace(_cfg["start"], f"OBS_REACH_CONFIGS[{_v!r}].start")   # type: ignore[arg-type]
    check_workspace(_cfg["goal"], f"OBS_REACH_CONFIGS[{_v!r}].goal")     # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Slalom obstacle geometry (T2, T4, T6)
# ---------------------------------------------------------------------------
#: Half-sizes of one slalom obstacle: a 6 cm x 6 cm x 30 cm upright bar,
#: matching the real-obstacle reach env's bar.
OBSTACLE_HALF_SIZES: Vec3 = (0.03, 0.03, 0.15)

#: Along-path and across-path offsets of the two flanking bars from the path
#: midpoint. Frozen with the layouts.
OBSTACLE_ALONG_PATH_OFFSET_M: float = 0.06
OBSTACLE_ACROSS_PATH_OFFSET_M: float = 0.08

#: Height the pick-and-place slalom path is evaluated at (the carried cube's
#: transport height), in the active frame's z -- unchanged by the frame shift.
OBSTACLE_PP_PATH_HEIGHT_M: float = 0.20


def slalom_obstacle_positions(
    start: Vec3,
    goal: Vec3,
    *,
    obs_half_height: float = OBSTACLE_HALF_SIZES[2],
) -> list[Vec3]:
    """Return the three obstacle centres for a start->goal slalom.

    obs0 sits on the path midpoint (blocking the direct route); obs1 and obs2
    straddle it, offset forward-left and backward-right in the XY plane. Every
    input is frozen per env, so the output is fully deterministic.
    """
    s = np.asarray(start, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    mid = (s + g) / 2.0

    diff_xy = g[:2] - s[:2]
    norm_xy = float(np.linalg.norm(diff_xy))
    # A purely vertical path has no XY direction; fall back to +X so the bars
    # still straddle the ascent instead of collapsing onto one point.
    path_dir = diff_xy / norm_xy if norm_xy >= 1e-6 else np.array([1.0, 0.0])
    perp_dir = np.array([-path_dir[1], path_dir[0]])  # left perpendicular

    along = OBSTACLE_ALONG_PATH_OFFSET_M
    across = OBSTACLE_ACROSS_PATH_OFFSET_M
    obs_z = float(obs_half_height)
    return [
        (float(mid[0]), float(mid[1]), obs_z),
        (
            float(mid[0] + path_dir[0] * along + perp_dir[0] * across),
            float(mid[1] + path_dir[1] * along + perp_dir[1] * across),
            obs_z,
        ),
        (
            float(mid[0] - path_dir[0] * along - perp_dir[0] * across),
            float(mid[1] - path_dir[1] * along - perp_dir[1] * across),
            obs_z,
        ),
    ]


# ---------------------------------------------------------------------------
# T9 / T10 -- Cluttered YCB layouts
# All object z values are per-model resting heights; distances are hand-checked
# to keep every pair >= 2 cm apart.
# ---------------------------------------------------------------------------
_M1_CLUTTERED_LAYOUTS: dict[str, dict] = {
    "v1": {
        "target": {"model": "025_mug", "pos": (-0.280, 0.050, 0.050), "yaw_deg": 0},
        "clutter": [
            {"model": "024_bowl", "pos": (-0.190, 0.200, 0.040), "yaw_deg": 0},
            {"model": "005_tomato_soup_can", "pos": (-0.360, 0.160, 0.040), "yaw_deg": 45},
            {"model": "009_gelatin_box", "pos": (-0.220, -0.150, 0.040), "yaw_deg": 30},
        ],
        "place_goal": (-0.380, 0.000, 0.035),
        "label": "loose_easy",
    },
    "v2": {
        "target": {"model": "006_mustard_bottle", "pos": (-0.250, 0.000, 0.060), "yaw_deg": 0},
        "clutter": [
            {"model": "003_cracker_box", "pos": (-0.200, 0.090, 0.060), "yaw_deg": 90},
            {"model": "004_sugar_box", "pos": (-0.200, -0.090, 0.050), "yaw_deg": 45},
            {"model": "010_potted_meat_can", "pos": (-0.310, 0.090, 0.040), "yaw_deg": 0},
            {"model": "009_gelatin_box", "pos": (-0.310, -0.090, 0.040), "yaw_deg": 60},
        ],
        "place_goal": (-0.390, 0.000, 0.035),
        "label": "dense_cluster",
    },
    "v3": {
        "target": {"model": "011_banana", "pos": (-0.300, -0.150, 0.040), "yaw_deg": 0},
        "clutter": [
            {"model": "025_mug", "pos": (-0.220, 0.000, 0.050), "yaw_deg": 0},
            {"model": "024_bowl", "pos": (-0.250, 0.150, 0.040), "yaw_deg": 0},
            {"model": "004_sugar_box", "pos": (-0.350, 0.100, 0.050), "yaw_deg": 30},
            {"model": "005_tomato_soup_can", "pos": (-0.380, -0.050, 0.040), "yaw_deg": 0},
            {"model": "009_gelatin_box", "pos": (-0.360, -0.200, 0.040), "yaw_deg": 45},
        ],
        "place_goal": (-0.180, 0.200, 0.035),
        "label": "arc_of_clutter",
    },
    "v4": {
        "target": {"model": "005_tomato_soup_can", "pos": (-0.230, 0.200, 0.040), "yaw_deg": 0},
        "clutter": [
            {"model": "024_bowl", "pos": (-0.280, 0.080, 0.040), "yaw_deg": 0},
            {"model": "006_mustard_bottle", "pos": (-0.200, 0.060, 0.060), "yaw_deg": 0},
            {"model": "003_cracker_box", "pos": (-0.340, 0.180, 0.060), "yaw_deg": 90},
        ],
        "place_goal": (-0.380, 0.000, 0.035),
        "label": "mixed_sizes_path_blocked",
    },
    "v5": {
        "target": {"model": "009_gelatin_box", "pos": (-0.280, 0.000, 0.040), "yaw_deg": 0},
        "clutter": [
            {"model": "025_mug", "pos": (-0.200, 0.100, 0.050), "yaw_deg": 0},
            {"model": "024_bowl", "pos": (-0.200, -0.100, 0.040), "yaw_deg": 0},
            {"model": "006_mustard_bottle", "pos": (-0.360, 0.100, 0.060), "yaw_deg": 0},
            {"model": "005_tomato_soup_can", "pos": (-0.360, -0.100, 0.040), "yaw_deg": 0},
            {"model": "003_cracker_box", "pos": (-0.220, 0.220, 0.060), "yaw_deg": 90},
            {"model": "004_sugar_box", "pos": (-0.220, -0.220, 0.050), "yaw_deg": 45},
            {"model": "010_potted_meat_can", "pos": (-0.340, 0.000, 0.040), "yaw_deg": 0},
            {"model": "011_banana", "pos": (-0.170, 0.000, 0.040), "yaw_deg": 0},
        ],
        "place_goal": (-0.400, 0.000, 0.035),
        "label": "maximum_clutter",
    },
}


def _layout_to_world(layout: dict) -> dict:
    return {
        "target": {**layout["target"], "pos": to_world(layout["target"]["pos"])},
        "clutter": [{**item, "pos": to_world(item["pos"])} for item in layout["clutter"]],
        "place_goal": to_world(layout["place_goal"]),
        "label": layout["label"],
    }


CLUTTERED_LAYOUTS: dict[str, dict] = {
    key: _layout_to_world(layout) for key, layout in _M1_CLUTTERED_LAYOUTS.items()
}

for _v, _layout in CLUTTERED_LAYOUTS.items():
    check_workspace(_layout["target"]["pos"], f"CLUTTERED[{_v!r}].target", axes="xy")
    for _index, _item in enumerate(_layout["clutter"]):
        check_workspace(_item["pos"], f"CLUTTERED[{_v!r}].clutter[{_index}]", axes="xy")
    check_workspace(_layout["place_goal"], f"CLUTTERED[{_v!r}].place_goal", axes="xy")


# ---------------------------------------------------------------------------
# Workspace wireframe (debug visualisation only)
# ---------------------------------------------------------------------------
def workspace_box_edges() -> list[dict]:
    """Return 12 edge descriptors for a wireframe of EVAL_WORKSPACE_BOUNDS.

    Each descriptor carries ``centre`` and ``half_sizes`` for
    ``actors.build_box``. Bars have a 2 mm square cross-section.
    """
    (x0, x1), (y0, y1), (z0, z1) = (tuple(row) for row in EVAL_WORKSPACE_BOUNDS)
    hx, hy, hz = (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    t = 0.002

    edges: list[dict] = []
    for yc in (y0, y1):
        for zc in (z0, z1):
            edges.append({"centre": (cx, yc, zc), "half_sizes": (hx, t, t)})
    for xc in (x0, x1):
        for zc in (z0, z1):
            edges.append({"centre": (xc, cy, zc), "half_sizes": (t, hy, t)})
    for xc in (x0, x1):
        for yc in (y0, y1):
            edges.append({"centre": (xc, yc, cz), "half_sizes": (t, t, hz)})
    return edges


if _OUT_OF_REACH_BOX:
    warnings.warn(
        f"{len(_OUT_OF_REACH_BOX)} frozen eval positions sit outside the conservative "
        "IK-verified reach box (inside the crop box and the training sampling box). "
        "See eval_config.workspace_report() for the list; this is expected for this "
        "frozen population.",
        stacklevel=2,
    )
