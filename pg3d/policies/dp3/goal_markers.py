from __future__ import annotations

import numpy as np

Array = np.ndarray

# Match the dataset bake (`--goal-marker-points 192 --goal-marker-radius 0.055`).
# The encoder splits off the trailing `goal_marker_points` slots as the goal
# branch, so this MUST equal the number of goal slots baked by the dataset
# writer or baked markers leak into the PointNet scene branch.
DEFAULT_GOAL_MARKER_POINTS = 192
DEFAULT_GOAL_MARKER_RADIUS = 0.055


def goal_marker_offsets(
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Return deterministic structured offsets used for target-centered goal tokens."""
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    r = np.float32(radius)
    if r == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    pattern: list[np.ndarray] = [np.zeros(3, dtype=np.float32)]
    cross_dirs = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    for direction in cross_dirs:
        pattern.append(r * direction)

    ring_count = max(0, num_points - len(pattern))
    for idx in range(ring_count):
        angle = 2.0 * np.pi * idx / max(ring_count, 1)
        ring_radius = r * (0.70 if idx % 2 == 0 else 1.00)
        z_offset = r * 0.25 * (1.0 if idx % 4 in {0, 1} else -1.0)
        pattern.append(
            np.asarray(
                [ring_radius * np.cos(angle), ring_radius * np.sin(angle), z_offset],
                dtype=np.float32,
            )
        )

    if num_points <= len(pattern):
        return np.asarray(pattern[:num_points], dtype=np.float32)
    repeats = int(np.ceil(num_points / len(pattern)))
    return np.tile(np.asarray(pattern, dtype=np.float32), (repeats, 1))[:num_points]


def goal_marker_points(
    target_position: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Return fixed ordered marker points centered at each target position."""
    target = np.asarray(target_position, dtype=np.float32)
    if target.shape[-1:] != (3,):
        raise ValueError(f"target_position must end with shape [3], got {target.shape}")
    offsets = goal_marker_offsets(num_points=num_points, radius=radius)
    if num_points == 0:
        return np.zeros((*target.shape[:-1], 0, 3), dtype=np.float32)
    return target[..., None, :] + offsets.reshape((1,) * (target.ndim - 1) + offsets.shape)


# Marker styles supported by the dataset writer and inference-time baking.
# ``sphere`` is the legacy rotation-symmetric, position-only marker.
# ``triad`` is an oriented coordinate-frame marker that encodes the full 6D
# goal pose (position + orientation) with no rotational symmetry.
GOAL_MARKER_STYLES = ("sphere", "triad")
DEFAULT_GOAL_MARKER_STYLE = "triad"

# Canonical triad arm lengths as multiples of ``radius``. The three arms have
# DISTINCT lengths and are drawn ONE-SIDED (only the + direction), so the
# marker has no rotational symmetry: any nonzero rotation (including a 180deg
# roll about any axis) maps it to a different ordered point set. This lets the
# encoder's ordered goal MLP read the full orientation unambiguously.
_TRIAD_X_LEN = 1.00  # longest arm
_TRIAD_Y_LEN = 0.60  # medium arm
_TRIAD_Z_LEN = 0.35  # shortest arm


def _quat_to_rotation_matrix(quat: Array) -> Array:
    """Convert a SAPIEN ``[w, x, y, z]`` quaternion to a 3x3 rotation matrix.

    Uses the SAPIEN/ManiSkill scalar-first convention to match the dataset
    writer (which stores TCP/goal poses as ``[w, x, y, z]``).
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = (q / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _allocate_triad_counts(num_points: int) -> tuple[int, int, int, int]:
    """Split ``num_points`` into (origin, x_arm, y_arm, z_arm) point counts.

    The origin cluster marks the goal position; the three arms encode
    orientation. Longer arms receive proportionally more points so the ordered
    marker degrades gracefully at small budgets. Guarantees the four counts sum
    to exactly ``num_points`` and that every arm gets at least one point when
    the budget allows.
    """
    if num_points <= 0:
        return 0, 0, 0, 0
    if num_points < 4:
        # Too few slots for a full frame: put everything in the origin cluster.
        return num_points, 0, 0, 0
    origin = max(1, int(round(num_points * 0.18)))
    remaining = num_points - origin
    arm_weights = np.asarray([_TRIAD_X_LEN, _TRIAD_Y_LEN, _TRIAD_Z_LEN], dtype=np.float64)
    arm_weights = arm_weights / float(arm_weights.sum())
    x_arm = max(1, int(np.floor(remaining * arm_weights[0])))
    y_arm = max(1, int(np.floor(remaining * arm_weights[1])))
    z_arm = max(1, remaining - x_arm - y_arm)
    # Correct any rounding drift so the total is exact.
    drift = num_points - (origin + x_arm + y_arm + z_arm)
    x_arm += drift
    return origin, x_arm, y_arm, z_arm


def _quats_to_rotation_matrices(quats: Array) -> Array:
    """Vectorized SAPIEN ``[..., 4]`` (w,x,y,z) -> ``[..., 3, 3]`` rotation matrices."""
    q = np.asarray(quats, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternions must end with shape [4], got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("quaternion must be non-zero")
    q = q / norm
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[..., 0, 1] = 2.0 * (x * y - w * z)
    m[..., 0, 2] = 2.0 * (x * z + w * y)
    m[..., 1, 0] = 2.0 * (x * y + w * z)
    m[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[..., 1, 2] = 2.0 * (y * z - w * x)
    m[..., 2, 0] = 2.0 * (x * z - w * y)
    m[..., 2, 1] = 2.0 * (y * z + w * x)
    m[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def _canonical_triad_offsets(num_points: int, radius: float) -> Array:
    """Return the unrotated ``[num_points, 3]`` canonical triad offsets."""
    origin_n, x_n, y_n, z_n = _allocate_triad_counts(num_points)
    r = float(radius)
    offsets: list[np.ndarray] = []
    if origin_n > 0:
        idx = np.arange(origin_n, dtype=np.float64)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - 2.0 * (idx + 0.5) / float(origin_n)
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        theta = idx * golden_angle
        shell = np.stack([radial * np.cos(theta), radial * np.sin(theta), z], axis=1)
        offsets.append((0.12 * r) * shell)

    def _arm(axis: np.ndarray, length: float, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float64)
        fracs = (np.arange(count, dtype=np.float64) + 1.0) / float(count)
        return np.outer(fracs * (length * r), axis)

    offsets.append(_arm(np.asarray([1.0, 0.0, 0.0]), _TRIAD_X_LEN, x_n))
    offsets.append(_arm(np.asarray([0.0, 1.0, 0.0]), _TRIAD_Y_LEN, y_n))
    offsets.append(_arm(np.asarray([0.0, 0.0, 1.0]), _TRIAD_Z_LEN, z_n))
    canonical = np.concatenate([o for o in offsets if o.size], axis=0)
    if canonical.shape[0] != num_points:
        if canonical.shape[0] < num_points:
            pad = np.repeat(canonical[-1:], num_points - canonical.shape[0], axis=0)
            canonical = np.concatenate([canonical, pad], axis=0)
        else:
            canonical = canonical[:num_points]
    return canonical


def triad_goal_marker(
    target_position: Array,
    quat: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Return ordered oriented-triad marker points centered at ``target_position``.

    ``target_position`` may be batched (shape ``[..., 3]``). ``quat`` (SAPIEN
    ``[w, x, y, z]``) may be either a single quaternion shared across the batch,
    or a per-row array whose leading dims match ``target_position[..., :-1]``
    (used when baking many episodes with different goal orientations at once).
    """
    target = np.asarray(target_position, dtype=np.float32)
    if target.shape[-1:] != (3,):
        raise ValueError(f"target_position must end with shape [3], got {target.shape}")
    if num_points == 0:
        return np.zeros((*target.shape[:-1], 0, 3), dtype=np.float32)
    if radius == 0:
        return np.zeros((*target.shape[:-1], num_points, 3), dtype=np.float32)

    canonical = _canonical_triad_offsets(num_points, radius)  # [P, 3]
    q = np.asarray(quat, dtype=np.float64)
    batch_shape = target.shape[:-1]

    if q.shape == (4,) or q.ndim == 1:
        rot = _quat_to_rotation_matrix(q)  # [3, 3]
        rotated = (canonical @ rot.T).astype(np.float32)  # [P, 3]
        marker = target[..., None, :] + rotated.reshape((1,) * len(batch_shape) + rotated.shape)
        return marker.astype(np.float32, copy=False)

    # Per-row quaternions: rotate the canonical offsets independently per row.
    if q.shape[:-1] != batch_shape:
        raise ValueError(
            f"quat batch shape {q.shape[:-1]} must match target batch shape {batch_shape}"
        )
    rots = _quats_to_rotation_matrices(q)  # [*batch, 3, 3]
    # [*batch, P, 3] = canonical[P,3] @ rots^T per row
    rotated = np.einsum("...ij,pj->...pi", rots, canonical).astype(np.float32)
    marker = target[..., None, :] + rotated
    return marker.astype(np.float32, copy=False)


def triad_goal_marker_offsets(
    quat: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Return ordered, goal-centered offsets for a single oriented-frame marker.

    Convenience wrapper returning just the rotated canonical offsets (no
    translation) for a single quaternion; used by tests and single-pose bakes.
    """
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if radius == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    canonical = _canonical_triad_offsets(num_points, radius)
    rot = _quat_to_rotation_matrix(quat)
    return (canonical @ rot.T).astype(np.float32, copy=False)


def build_goal_marker(
    target_position: Array,
    quat: Array | None = None,
    *,
    style: str = DEFAULT_GOAL_MARKER_STYLE,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
) -> Array:
    """Dispatch marker construction by ``style``.

    ``sphere`` ignores ``quat`` (position-only). ``triad`` requires ``quat`` and
    encodes the full 6D pose. This is the single shared entry point used by both
    the dataset writer and inference-time baking so they never diverge.
    """
    if style == "sphere":
        return goal_marker_points(target_position, num_points=num_points, radius=radius)
    if style == "triad":
        if quat is None:
            raise ValueError("triad goal marker requires a quaternion")
        return triad_goal_marker(
            target_position, quat, num_points=num_points, radius=radius
        )
    raise ValueError(
        f"unsupported goal marker style {style!r}; expected one of {GOAL_MARKER_STYLES}"
    )


def insert_goal_marker_points(
    point_cloud: Array,
    target_position: Array,
    *,
    num_points: int = DEFAULT_GOAL_MARKER_POINTS,
    radius: float = DEFAULT_GOAL_MARKER_RADIUS,
    style: str = "sphere",
    quat: Array | None = None,
) -> Array:
    """Overwrite the final ``num_points`` point-cloud slots with ordered goal tokens.

    ``style`` selects the marker geometry (``sphere`` position-only, default for
    backward compatibility, or ``triad`` which requires ``quat`` and encodes the
    full 6D goal pose).
    """
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.shape[-1:] != (3,):
        raise ValueError(f"point_cloud must end with shape [*, 3], got {points.shape}")
    if points.ndim < 2:
        raise ValueError(f"point_cloud must have at least 2 dimensions, got {points.shape}")
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if num_points == 0:
        return points.astype(np.float32, copy=True)
    if num_points >= points.shape[-2]:
        raise ValueError(
            "num_points must be smaller than the point-cloud point count "
            f"({num_points} >= {points.shape[-2]})"
        )

    marker = build_goal_marker(
        target_position,
        quat,
        style=style,
        num_points=num_points,
        radius=radius,
    )
    expected_marker_shape = (*points.shape[:-2], num_points, 3)
    try:
        marker = np.broadcast_to(marker, expected_marker_shape)
    except ValueError as exc:
        raise ValueError(
            f"target_position shape {np.asarray(target_position).shape} cannot broadcast "
            f"to point_cloud shape {points.shape}"
        ) from exc

    output = points.astype(np.float32, copy=True)
    output[..., -num_points:, :] = marker
    return output
