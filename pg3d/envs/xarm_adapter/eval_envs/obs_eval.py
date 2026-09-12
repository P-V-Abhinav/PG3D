"""obs_eval.py -- Deterministic obstacle eval envs (T2, T4, T6).

Two families:

  ``PG3DReach-Eval-Obs-Reach-v*``  reach through a three-bar slalom      (T2)
  ``PG3DReach-Eval-Obs-PP-v*``     pick and place through a slalom  (T4, T6)

Obstacle geometry is one frozen shape -- three upright 6 x 6 x 30 cm bars -- and
their positions are a pure function of the frozen start and goal (see
``eval_config.slalom_obstacle_positions``): one bar on the path midpoint and two
straddling it. Nothing is sampled, so the layout is identical on every run and
every machine.

The bars are real scene geometry, so unlike the goal/start markers they are NOT
blanked while observations render: the policy is meant to see them in the point
cloud.

They are also COLLIDABLE. The suite this was ported from built them with
``add_collision=False``, which makes an "avoidance" task unfalsifiable -- the arm
can sweep straight through a bar and still be graded successful. Physical
contact is what T2/T4/T6 grade against, so collision is on by default; pass
``obstacle_collision=False`` to reproduce the old virtual-obstacle behaviour.
"""

from __future__ import annotations

from typing import Any

import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .eval_base import PG3DEvalBase
from .eval_config import (
    CUBE_HALF_SIZE,
    CUBE_POSITIONS,
    OBS_PP_START_QPOS,
    OBS_PP_START_TCPS,
    OBS_REACH_CONFIGS,
    OBS_REACH_START_QPOS,
    OBSTACLE_HALF_SIZES,
    OBSTACLE_PP_PATH_HEIGHT_M,
    PLACE_TARGETS,
    slalom_obstacle_positions,
)
from .pp_eval import PG3DEvalPickPlaceEnv

Vec3 = tuple[float, float, float]


class _BarObstacleMixin:
    """Build and place upright cuboid bars at frozen positions.

    One bar shape for the whole suite -- 6 x 6 x 30 cm, matching the
    real-obstacle reach env -- so every obstacle scene is the same primitive
    repeated. Subclasses say WHERE via :meth:`obstacle_positions`; a slalom
    computes them from a path, a wall or a U-trap lists them out.

    Bars are collidable by default: an "avoidance" task whose obstacles are
    ghosts cannot be failed. Pass ``obstacle_collision=False`` for the old
    virtual-obstacle behaviour.
    """

    OBSTACLE_HALF_SIZES: Vec3 = OBSTACLE_HALF_SIZES

    def __init__(self, *args: Any, obstacle_collision: bool = True, **kwargs: Any) -> None:
        self._obstacle_collision = bool(obstacle_collision)
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    @classmethod
    def obstacle_positions(cls) -> list[Vec3]:
        """Frozen bar centres, computable without building the env."""
        raise NotImplementedError

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)  # type: ignore[misc]
        self.obstacle_actors: list[Any] = []
        for index, position in enumerate(self.obstacle_positions()):
            self.obstacle_actors.append(
                actors.build_box(
                    self.scene,  # type: ignore[attr-defined]
                    half_sizes=list(self.OBSTACLE_HALF_SIZES),
                    color=[0.10, 0.10, 0.90, 1.0],
                    name=f"obstacle_{index}",
                    body_type="kinematic",
                    add_collision=self._obstacle_collision,
                    initial_pose=sapien.Pose(p=list(position)),
                )
            )

    def _place_obstacles(self, batch: int) -> None:
        for actor, position in zip(self.obstacle_actors, self.obstacle_positions(), strict=True):
            pose = torch.tensor(
                [list(position)],
                dtype=torch.float32,
                device=self.device,  # type: ignore[attr-defined]
            ).expand(batch, -1)
            actor.set_pose(Pose.create_from_pq(pose))


class _SlalomObstacleMixin(_BarObstacleMixin):
    """Bars placed as a slalom across a start->goal path.

    Subclasses implement :meth:`slalom_endpoints` to say which path the slalom
    straddles: the TCP path for a reach, the carried-cube transport path for a
    pick and place.
    """

    @classmethod
    def slalom_endpoints(cls) -> tuple[Vec3, Vec3]:
        raise NotImplementedError

    @classmethod
    def obstacle_positions(cls) -> list[Vec3]:
        """The three frozen bar centres for this variant's start->goal path."""
        start, goal = cls.slalom_endpoints()
        return slalom_obstacle_positions(start, goal, obs_half_height=cls.OBSTACLE_HALF_SIZES[2])


# ---------------------------------------------------------------------------
# T2 -- reach through the slalom
# ---------------------------------------------------------------------------
class PG3DEvalObsReachEnv(_SlalomObstacleMixin, PG3DEvalBase):
    """Reach a frozen goal from the frozen start without touching three bars."""

    TASK_IDS = ("T2",)

    @classmethod
    def slalom_endpoints(cls) -> tuple[Vec3, Vec3]:
        return cls.START_TCP_POS, cls.GOAL_POS

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        self._place_obstacles(len(env_idx))


@register_env("PG3DReach-Eval-Obs-Reach-v1", max_episode_steps=200)
class PG3DEvalObsReachV1(PG3DEvalObsReachEnv):
    """Obs-Reach v1 -- forward-left slalom."""

    GOAL_POS = OBS_REACH_CONFIGS["v1"]["goal"]
    START_TCP_POS = OBS_REACH_CONFIGS["v1"]["start"]
    START_QPOS = OBS_REACH_START_QPOS["v1"]


@register_env("PG3DReach-Eval-Obs-Reach-v2", max_episode_steps=200)
class PG3DEvalObsReachV2(PG3DEvalObsReachEnv):
    """Obs-Reach v2 -- pure forward slalom."""

    GOAL_POS = OBS_REACH_CONFIGS["v2"]["goal"]
    START_TCP_POS = OBS_REACH_CONFIGS["v2"]["start"]
    START_QPOS = OBS_REACH_START_QPOS["v2"]


@register_env("PG3DReach-Eval-Obs-Reach-v3", max_episode_steps=200)
class PG3DEvalObsReachV3(PG3DEvalObsReachEnv):
    """Obs-Reach v3 -- forward-right slalom."""

    GOAL_POS = OBS_REACH_CONFIGS["v3"]["goal"]
    START_TCP_POS = OBS_REACH_CONFIGS["v3"]["start"]
    START_QPOS = OBS_REACH_START_QPOS["v3"]


@register_env("PG3DReach-Eval-Obs-Reach-v4", max_episode_steps=200)
class PG3DEvalObsReachV4(PG3DEvalObsReachEnv):
    """Obs-Reach v4 -- backward-left slalom."""

    GOAL_POS = OBS_REACH_CONFIGS["v4"]["goal"]
    START_TCP_POS = OBS_REACH_CONFIGS["v4"]["start"]
    START_QPOS = OBS_REACH_START_QPOS["v4"]


@register_env("PG3DReach-Eval-Obs-Reach-v5", max_episode_steps=200)
class PG3DEvalObsReachV5(PG3DEvalObsReachEnv):
    """Obs-Reach v5 -- backward-right slalom."""

    GOAL_POS = OBS_REACH_CONFIGS["v5"]["goal"]
    START_TCP_POS = OBS_REACH_CONFIGS["v5"]["start"]
    START_QPOS = OBS_REACH_START_QPOS["v5"]


# ---------------------------------------------------------------------------
# T4 / T6 -- pick and place through the slalom
#
# The slalom straddles the carried-cube transport path: from above the cube to
# above the place target, both at OBSTACLE_PP_PATH_HEIGHT_M. Cube and place
# target are the canonical ones shared with the plain pick-and-place envs, so a
# T3-vs-T4 comparison differs only by the obstacles.
# ---------------------------------------------------------------------------
class PG3DEvalObsPickPlaceEnv(_SlalomObstacleMixin, PG3DEvalPickPlaceEnv):
    """Pick and place through the frozen slalom."""

    TASK_IDS = ("T4", "T6")

    CUBE_HALF_SIZE: float = CUBE_HALF_SIZE

    @classmethod
    def slalom_endpoints(cls) -> tuple[Vec3, Vec3]:
        height = OBSTACLE_PP_PATH_HEIGHT_M
        return (
            (cls.CUBE_POS[0], cls.CUBE_POS[1], height),
            (cls.GOAL_POS[0], cls.GOAL_POS[1], height),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        self._place_obstacles(len(env_idx))


@register_env("PG3DReach-Eval-Obs-PP-v1", max_episode_steps=250)
class PG3DEvalObsPPV1(PG3DEvalObsPickPlaceEnv):
    """Obs-PP v1 -- lateral transport slalom."""

    CUBE_POS = CUBE_POSITIONS["v1"]
    GOAL_POS = PLACE_TARGETS["v1"]
    START_TCP_POS = OBS_PP_START_TCPS["v1"]
    START_QPOS = OBS_PP_START_QPOS["v1"]


@register_env("PG3DReach-Eval-Obs-PP-v2", max_episode_steps=250)
class PG3DEvalObsPPV2(PG3DEvalObsPickPlaceEnv):
    """Obs-PP v2 -- longitudinal pull-back slalom."""

    CUBE_POS = CUBE_POSITIONS["v2"]
    GOAL_POS = PLACE_TARGETS["v2"]
    START_TCP_POS = OBS_PP_START_TCPS["v2"]
    START_QPOS = OBS_PP_START_QPOS["v2"]


@register_env("PG3DReach-Eval-Obs-PP-v3", max_episode_steps=250)
class PG3DEvalObsPPV3(PG3DEvalObsPickPlaceEnv):
    """Obs-PP v3 -- diagonal cross-sweep slalom."""

    CUBE_POS = CUBE_POSITIONS["v3"]
    GOAL_POS = PLACE_TARGETS["v3"]
    START_TCP_POS = OBS_PP_START_TCPS["v3"]
    START_QPOS = OBS_PP_START_QPOS["v3"]


@register_env("PG3DReach-Eval-Obs-PP-v4", max_episode_steps=250)
class PG3DEvalObsPPV4(PG3DEvalObsPickPlaceEnv):
    """Obs-PP v4 -- diagonal pull-back slalom."""

    CUBE_POS = CUBE_POSITIONS["v4"]
    GOAL_POS = PLACE_TARGETS["v4"]
    START_TCP_POS = OBS_PP_START_TCPS["v4"]
    START_QPOS = OBS_PP_START_QPOS["v4"]


@register_env("PG3DReach-Eval-Obs-PP-v5", max_episode_steps=250)
class PG3DEvalObsPPV5(PG3DEvalObsPickPlaceEnv):
    """Obs-PP v5 -- far-front-left diagonal slalom."""

    CUBE_POS = CUBE_POSITIONS["v5"]
    GOAL_POS = PLACE_TARGETS["v5"]
    START_TCP_POS = OBS_PP_START_TCPS["v5"]
    START_QPOS = OBS_PP_START_QPOS["v5"]
