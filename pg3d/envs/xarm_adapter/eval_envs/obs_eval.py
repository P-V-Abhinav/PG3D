"""obs_eval.py — Deterministic Obstacle evaluation environments (Tasks 3, 4, 6).

Two env families:
  PG3DEvalObsReachEnv  — reach through 3 slalom obstacles  (Task 3)
  PG3DEvalObsPPEnv     — pick & place through obstacles     (Tasks 4, 6)

Obstacle geometry
-----------------
Each obstacle is a tall kinematic blue cuboid:
  half_sizes = [0.03, 0.03, 0.15]  →  6 cm × 6 cm × 30 cm
  (matches PG3DReachRealObstacleEnv in obstacle_envs.py)

Obstacle placement (deterministic)
-----------------------------------
Given a hardcoded start and goal position (TCP targets in world frame),
the three obstacles are placed as a slalom on the direct path:

  obs0  : path midpoint                              (blocks direct route)
  obs1  : midpoint + path_dir * 0.06 + perp * 0.08  (forward-left)
  obs2  : midpoint - path_dir * 0.06 - perp * 0.08  (back-right)

Because start and goal are class-level constants, all obstacle positions
are fully deterministic — no randomness.

Task 6 (L/R + Avoidance) reuses PG3DReach-Eval-Obs-PP-v* directly.
The L/R distinction is an ApproachPostureConstraint in the eval script.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .eval_config import CUBE_POSITIONS, OBS_REACH_CONFIGS, PLACE_TARGETS
from .reach_eval import PG3DEvalBase


# ---------------------------------------------------------------------------
# Shared: obstacle placement utility
# ---------------------------------------------------------------------------

def _slalom_obstacle_positions(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    *,
    obs_half_height: float = 0.15,
) -> list[tuple[float, float, float]]:
    """Compute the 3 obstacle centres for a start→goal slalom path.

    Returns a list of 3 (x, y, z) world-frame positions.
    Obstacles are placed at height z = obs_half_height (their centre).
    """
    s = np.array(start, dtype=np.float64)
    g = np.array(goal,  dtype=np.float64)
    mid = (s + g) / 2.0

    # Forward unit vector in the XY plane
    diff_xy = g[:2] - s[:2]
    norm_xy = np.linalg.norm(diff_xy)
    if norm_xy < 1e-6:
        # Path is vertical — place obstacles in a fixed pattern
        path_dir = np.array([1.0, 0.0])
    else:
        path_dir = diff_xy / norm_xy

    perp_dir = np.array([-path_dir[1], path_dir[0]])  # left perpendicular

    obs_z = obs_half_height

    positions = [
        (float(mid[0]),
         float(mid[1]),
         obs_z),

        (float(mid[0] + path_dir[0] * 0.06 + perp_dir[0] * 0.08),
         float(mid[1] + path_dir[1] * 0.06 + perp_dir[1] * 0.08),
         obs_z),

        (float(mid[0] - path_dir[0] * 0.06 - perp_dir[0] * 0.08),
         float(mid[1] - path_dir[1] * 0.06 - perp_dir[1] * 0.08),
         obs_z),
    ]
    return positions


# ---------------------------------------------------------------------------
# Obstacle mixin — builds 3 tall cuboid obstacles
# ---------------------------------------------------------------------------

class _ObstacleMixin:
    """Mixin that adds 3 kinematic tall-cuboid obstacles to any eval base.

    half_sizes=[0.03, 0.03, 0.15] → 6 cm × 6 cm × 30 cm blue cuboids.
    Subclass must provide START_POS and GOAL_POS (or CUBE_POS / PLACE_POS).
    """

    # Half-sizes matching PG3DReachRealObstacleEnv
    OBS_HALF_SIZES: list[float] = [0.03, 0.03, 0.15]

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)   # type: ignore[misc]
        self._obstacles: list[Any] = []
        for i in range(3):
            obs = actors.build_box(
                self.scene,                         # type: ignore[attr-defined]
                half_sizes=self.OBS_HALF_SIZES,
                color=[0.10, 0.10, 0.90, 1.0],
                name=f"obstacle_{i}",
                body_type="kinematic",
                add_collision=False,
            )
            self._obstacles.append(obs)

    def _place_obstacles(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
    ) -> None:
        """Place the 3 obstacles on the start→goal slalom path."""
        positions = _slalom_obstacle_positions(
            start, goal,
            obs_half_height=self.OBS_HALF_SIZES[2],
        )
        for obs, pos in zip(self._obstacles, positions):
            pos_t = torch.tensor(
                [list(pos)],
                dtype=torch.float32,
                device=self.device,               # type: ignore[attr-defined]
            )
            obs.set_pose(Pose.create_from_pq(pos_t))


# ---------------------------------------------------------------------------
# ENV 3: Obstacle Reach  (Task 3)
# ---------------------------------------------------------------------------

class PG3DEvalObsReachEnv(_ObstacleMixin, PG3DEvalBase):
    """Deterministic reach-through-obstacles base.

    Subclasses set:
      START_POS : (x, y, z) — TCP at episode start (used for obstacle placement)
      GOAL_POS  : (x, y, z) — TCP reach target
    """

    START_POS: tuple[float, float, float]
    GOAL_POS:  tuple[float, float, float]

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            n = len(env_idx)
            goal_t = torch.tensor(
                [list(self.GOAL_POS)] * n,
                dtype=torch.float32,
                device=self.device,
            )
            self.goal_site.set_pose(Pose.create_from_pq(goal_t))
        self._place_obstacles(self.START_POS, self.GOAL_POS)


@register_env("PG3DReach-Eval-Obs-Reach-v1", max_episode_steps=200)
class PG3DEvalObsReachV1(PG3DEvalObsReachEnv):
    """Obs-Reach v1 — pure lateral slalom."""
    START_POS = OBS_REACH_CONFIGS["v1"]["start"]
    GOAL_POS  = OBS_REACH_CONFIGS["v1"]["goal"]


@register_env("PG3DReach-Eval-Obs-Reach-v2", max_episode_steps=200)
class PG3DEvalObsReachV2(PG3DEvalObsReachEnv):
    """Obs-Reach v2 — pure forward slalom."""
    START_POS = OBS_REACH_CONFIGS["v2"]["start"]
    GOAL_POS  = OBS_REACH_CONFIGS["v2"]["goal"]


@register_env("PG3DReach-Eval-Obs-Reach-v3", max_episode_steps=200)
class PG3DEvalObsReachV3(PG3DEvalObsReachEnv):
    """Obs-Reach v3 — vertical slalom."""
    START_POS = OBS_REACH_CONFIGS["v3"]["start"]
    GOAL_POS  = OBS_REACH_CONFIGS["v3"]["goal"]


@register_env("PG3DReach-Eval-Obs-Reach-v4", max_episode_steps=200)
class PG3DEvalObsReachV4(PG3DEvalObsReachEnv):
    """Obs-Reach v4 — 3D diagonal slalom."""
    START_POS = OBS_REACH_CONFIGS["v4"]["start"]
    GOAL_POS  = OBS_REACH_CONFIGS["v4"]["goal"]


@register_env("PG3DReach-Eval-Obs-Reach-v5", max_episode_steps=200)
class PG3DEvalObsReachV5(PG3DEvalObsReachEnv):
    """Obs-Reach v5 — dense forward slalom (tightest)."""
    START_POS = OBS_REACH_CONFIGS["v5"]["start"]
    GOAL_POS  = OBS_REACH_CONFIGS["v5"]["goal"]


# ---------------------------------------------------------------------------
# ENV 4: Obstacle Pick & Place  (Tasks 4, 6)
#
# Cube position and place target use the canonical CUBE_POSITIONS /
# PLACE_TARGETS shared with the plain PP envs.
#
# The slalom path is computed between:
#   slalom_start = (cube_x, cube_y, 0.20)   ← lifted above cube
#   slalom_goal  = (place_x, place_y, 0.20) ← above place target
#
# This gives a horizontal obstacle field at 0.15 m height that the
# arm must navigate while carrying the cube.
# ---------------------------------------------------------------------------

class PG3DEvalObsPPEnv(_ObstacleMixin, PG3DEvalBase):
    """Deterministic obstacle Pick & Place base.

    Subclasses set CUBE_POS and PLACE_POS from canonical config.
    """

    CUBE_POS:  tuple[float, float, float]
    PLACE_POS: tuple[float, float, float]
    CUBE_HALF_SIZE: float = 0.035

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        self.cube = actors.build_box(
            self.scene,
            half_sizes=[self.CUBE_HALF_SIZE] * 3,
            color=[0.85, 0.20, 0.20, 1.0],
            name="cube",
            body_type="dynamic",
        )

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            n = len(env_idx)

            # Place cube
            cube_t = torch.tensor(
                [list(self.CUBE_POS)] * n,
                dtype=torch.float32,
                device=self.device,
            )
            self.cube.set_pose(Pose.create_from_pq(cube_t))

            # Place goal_site at place target
            place_t = torch.tensor(
                [list(self.PLACE_POS)] * n,
                dtype=torch.float32,
                device=self.device,
            )
            self.goal_site.set_pose(Pose.create_from_pq(place_t))

        # Slalom path runs between cube and place target, both lifted to 0.20m
        slalom_start = (self.CUBE_POS[0],  self.CUBE_POS[1],  0.20)
        slalom_goal  = (self.PLACE_POS[0], self.PLACE_POS[1], 0.20)
        self._place_obstacles(slalom_start, slalom_goal)


@register_env("PG3DReach-Eval-Obs-PP-v1", max_episode_steps=250)
class PG3DEvalObsPPV1(PG3DEvalObsPPEnv):
    """Obs-PP v1 — lateral slalom (cube front-center)."""
    CUBE_POS  = CUBE_POSITIONS["v1"]
    PLACE_POS = PLACE_TARGETS["v1"]


@register_env("PG3DReach-Eval-Obs-PP-v2", max_episode_steps=250)
class PG3DEvalObsPPV2(PG3DEvalObsPPEnv):
    """Obs-PP v2 — forward slalom (far cube)."""
    CUBE_POS  = CUBE_POSITIONS["v2"]
    PLACE_POS = PLACE_TARGETS["v2"]


@register_env("PG3DReach-Eval-Obs-PP-v3", max_episode_steps=250)
class PG3DEvalObsPPV3(PG3DEvalObsPPEnv):
    """Obs-PP v3 — cross-sweep slalom."""
    CUBE_POS  = CUBE_POSITIONS["v3"]
    PLACE_POS = PLACE_TARGETS["v3"]


@register_env("PG3DReach-Eval-Obs-PP-v4", max_episode_steps=250)
class PG3DEvalObsPPV4(PG3DEvalObsPPEnv):
    """Obs-PP v4 — diagonal slalom."""
    CUBE_POS  = CUBE_POSITIONS["v4"]
    PLACE_POS = PLACE_TARGETS["v4"]


@register_env("PG3DReach-Eval-Obs-PP-v5", max_episode_steps=250)
class PG3DEvalObsPPV5(PG3DEvalObsPPEnv):
    """Obs-PP v5 — vertical-lift slalom (hardest)."""
    CUBE_POS  = CUBE_POSITIONS["v5"]
    PLACE_POS = PLACE_TARGETS["v5"]
