"""trap_eval.py -- Local-minimum obstacle envs (T2 family).

Two deterministic reach scenes built to separate a greedy reacher from one
guided by an imagined rollout. Both start the arm at the suite's default start
pose and put the goal straight ahead at y = 0, so the straight line from start
to goal runs into the obstacle:

  ``PG3DReach-Eval-UTrap-v1``   A three-sided cup whose mouth faces the arm.
      Following the goal direction parks the gripper against the back wall with
      the goal a few centimetres past it, and every escape (around a side wall,
      or over the 30 cm top) moves AWAY from the goal first. That first move is
      what a greedy step cannot take and what scoring imagined rollouts can.

  ``PG3DReach-Eval-Gap-v1``     Two blocks with a 6 cm slot between them, dead
      on the straight line. The slot looks like the shortest route and is far
      too narrow for the gripper (jaws alone span ~8.5 cm), so a method that
      reasons about the TCP as a point drives into it and wedges. Going around
      either block is longer and is the only thing that works.

Both are built from the SAME 6 x 6 x 30 cm bar as the slalom obstacles, butted
together into walls -- no new scene primitive. Bar centres, goals and the
properties that make each scene a trap are frozen and asserted in eval_config.

What these envs do NOT do is make the policy see the obstacle. The checkpoint's
own input is robot points plus the goal marker (the dataset's crop keeps only
robot-masked points), so the obstacle reaches the method through the constraint
an eval script derives from the scene cloud, and avoidance is the harness's job
by construction -- which is the claim these two scenes are built to test.
"""

from __future__ import annotations

from typing import Any

import torch
from mani_skill.utils.registration import register_env

from .eval_base import PG3DEvalBase
from .eval_config import (
    GAP_BAR_POSITIONS,
    GAP_GOAL,
    START_TCP,
    SUITE_START_QPOS,
    UTRAP_BAR_POSITIONS,
    UTRAP_GOAL,
)
from .obs_eval import _BarObstacleMixin

Vec3 = tuple[float, float, float]


class PG3DEvalTrapReachEnv(_BarObstacleMixin, PG3DEvalBase):
    """Reach a frozen goal whose straight-line approach is a local minimum."""

    TASK_IDS = ("T2",)

    #: Frozen bar centres, listed rather than derived from the path: the whole
    #: point of these scenes is a specific shape, not a shape that follows the
    #: start and goal around.
    BAR_POSITIONS: tuple[Vec3, ...] = ()

    START_TCP_POS = START_TCP
    START_QPOS = SUITE_START_QPOS

    @classmethod
    def obstacle_positions(cls) -> list[Vec3]:
        return [tuple(float(v) for v in p) for p in cls.BAR_POSITIONS]  # type: ignore[misc]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        self._place_obstacles(len(env_idx))


@register_env("PG3DReach-Eval-UTrap-v1", max_episode_steps=250)
class PG3DEvalUTrapV1(PG3DEvalTrapReachEnv):
    """U-trap -- cup mouth facing the arm, goal just beyond the back wall."""

    BAR_POSITIONS = UTRAP_BAR_POSITIONS
    GOAL_POS = UTRAP_GOAL


@register_env("PG3DReach-Eval-Gap-v1", max_episode_steps=250)
class PG3DEvalGapV1(PG3DEvalTrapReachEnv):
    """Narrow-gap wall -- a 6 cm slot on the straight line, too small to pass."""

    BAR_POSITIONS = GAP_BAR_POSITIONS
    GOAL_POS = GAP_GOAL
