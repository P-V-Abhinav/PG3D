"""reach_eval.py -- Deterministic reach eval envs (T1).

Five variants, one frozen goal each, no randomness in the task. The camera's
per-episode jitter and calibration-error model are deliberately KEPT: they are
part of the deployment observation model, not task randomness.

Everything the parent guarantees (start pose at ``START_TCP``, frozen goal,
sphere marker contract, virtual markers) is described in :mod:`eval_base`.
"""

from __future__ import annotations

from mani_skill.utils.registration import register_env

from .eval_base import PG3DEvalBase
from .eval_config import REACH_GOALS, REACH_START_QPOS, REACH_START_TCPS


class PG3DEvalReachEnv(PG3DEvalBase):
    """Reach one frozen TCP position from this variant's own frozen start."""

    TASK_IDS = ("T1",)


@register_env("PG3DReach-Eval-Reach-v1", max_episode_steps=150)
class PG3DEvalReachV1(PG3DEvalReachEnv):
    """Reach v1 -- far front, mid height."""

    GOAL_POS = REACH_GOALS["v1"]
    START_TCP_POS = REACH_START_TCPS["v1"]
    START_QPOS = REACH_START_QPOS["v1"]


@register_env("PG3DReach-Eval-Reach-v2", max_episode_steps=150)
class PG3DEvalReachV2(PG3DEvalReachEnv):
    """Reach v2 -- far front-left, close to the table surface."""

    GOAL_POS = REACH_GOALS["v2"]
    START_TCP_POS = REACH_START_TCPS["v2"]
    START_QPOS = REACH_START_QPOS["v2"]


@register_env("PG3DReach-Eval-Reach-v3", max_episode_steps=150)
class PG3DEvalReachV3(PG3DEvalReachEnv):
    """Reach v3 -- far left diagonal."""

    GOAL_POS = REACH_GOALS["v3"]
    START_TCP_POS = REACH_START_TCPS["v3"]
    START_QPOS = REACH_START_QPOS["v3"]


@register_env("PG3DReach-Eval-Reach-v4", max_episode_steps=150)
class PG3DEvalReachV4(PG3DEvalReachEnv):
    """Reach v4 -- far right diagonal."""

    GOAL_POS = REACH_GOALS["v4"]
    START_TCP_POS = REACH_START_TCPS["v4"]
    START_QPOS = REACH_START_QPOS["v4"]


@register_env("PG3DReach-Eval-Reach-v5", max_episode_steps=150)
class PG3DEvalReachV5(PG3DEvalReachEnv):
    """Reach v5 -- back and high (the tallest goal in the suite)."""

    GOAL_POS = REACH_GOALS["v5"]
    START_TCP_POS = REACH_START_TCPS["v5"]
    START_QPOS = REACH_START_QPOS["v5"]
