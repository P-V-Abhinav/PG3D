"""The frozen 25-environment PG3D evaluation suite for the ten-task ladder.

Every env in here is deterministic and self-contained: importing the package
registers the ids, and a registered id fixes the start pose, the policy goal,
the object/obstacle layout, the goal-marker contract, the camera and the arm.
An eval run therefore needs an env id and a checkpoint -- no ``--source
dataset``, no ``--episode-indices``, no CLI goal, no camera flags.

    from pg3d.envs.xarm_adapter.eval_envs import register_pg3d_eval_envs
    register_pg3d_eval_envs()
    env = gym.make("PG3DReach-Eval-Reach-v3", obs_mode="pointcloud", num_envs=1)
    spec = env.unwrapped.pg3d_eval_spec      # start, goal, layout, marker, budget

Suite layout (5 variants each), mapped to docs/evaluation_tasks_scratchpad.md:

    PG3DReach-Eval-Reach-v1..v5       T1  reach
    PG3DReach-Eval-Obs-Reach-v1..v5   T2  reach with obstacle avoidance
    PG3DReach-Eval-PP-v1..v5          T3  pick and place
                                      T5  pick from the left/right
                                      T7  pick and place at a given pose
                                      T8  maintain object orientation
    PG3DReach-Eval-Obs-PP-v1..v5      T4  pick and place with avoidance
                                      T6  left/right pick-place with avoidance
    PG3DReach-Eval-Cluttered-v1..v5   T9  pick and place in clutter
                                      T10 place relative to another object

The scene is what an env freezes; the task is that scene plus the constraints an
eval script binds, which is why one env backs several tasks.
"""

from __future__ import annotations

_REGISTERED = False

#: Env ids per family, in variant order.
EVAL_ENV_IDS: dict[str, tuple[str, ...]] = {
    "reach": tuple(f"PG3DReach-Eval-Reach-v{i}" for i in range(1, 6)),
    "obs_reach": tuple(f"PG3DReach-Eval-Obs-Reach-v{i}" for i in range(1, 6)),
    "pick_place": tuple(f"PG3DReach-Eval-PP-v{i}" for i in range(1, 6)),
    "obs_pick_place": tuple(f"PG3DReach-Eval-Obs-PP-v{i}" for i in range(1, 6)),
    "cluttered": tuple(f"PG3DReach-Eval-Cluttered-v{i}" for i in range(1, 6)),
}

#: Which env family serves which scratchpad task id.
TASK_ENV_FAMILIES: dict[str, str] = {
    "T1": "reach",
    "T2": "obs_reach",
    "T3": "pick_place",
    "T4": "obs_pick_place",
    "T5": "pick_place",
    "T6": "obs_pick_place",
    "T7": "pick_place",
    "T8": "pick_place",
    "T9": "cluttered",
    "T10": "cluttered",
}


def env_ids_for_task(task_id: str) -> tuple[str, ...]:
    """Return the five env ids that serve one task id (``"T1"``..``"T10"``)."""
    key = str(task_id).upper()
    if key not in TASK_ENV_FAMILIES:
        raise KeyError(f"unknown task id {task_id!r}; expected one of {sorted(TASK_ENV_FAMILIES)}")
    return EVAL_ENV_IDS[TASK_ENV_FAMILIES[key]]


def all_eval_env_ids() -> tuple[str, ...]:
    """Return all 25 env ids, family by family."""
    return tuple(env_id for family in EVAL_ENV_IDS.values() for env_id in family)


def register_pg3d_eval_envs() -> None:
    """Import every eval env module, which triggers its ``@register_env``."""
    global _REGISTERED
    if _REGISTERED:
        return
    from pg3d.envs.xarm_adapter.eval_envs import cluttered_eval  # noqa: F401
    from pg3d.envs.xarm_adapter.eval_envs import obs_eval  # noqa: F401
    from pg3d.envs.xarm_adapter.eval_envs import pp_eval  # noqa: F401
    from pg3d.envs.xarm_adapter.eval_envs import reach_eval  # noqa: F401

    _REGISTERED = True


def eval_spec_for_env(env):  # noqa: ANN001, ANN201 - thin re-export
    """Return the frozen :class:`EvalEnvSpec` for a live eval env."""
    from .eval_specs import eval_spec_for_env as _impl

    return _impl(env)


def env_episode_context(env, obs=None, info=None):  # noqa: ANN001, ANN201
    """Return the env-sourced replacement for a Zarr episode context."""
    from .eval_specs import env_episode_context as _impl

    return _impl(env, obs, info)


def task_spec_for_env(env, **kwargs):  # noqa: ANN001, ANN201
    """Return a master ``TaskSpec`` built from a live eval env."""
    from .eval_specs import task_spec_for_env as _impl

    return _impl(env, **kwargs)


__all__ = [
    "EVAL_ENV_IDS",
    "TASK_ENV_FAMILIES",
    "all_eval_env_ids",
    "env_episode_context",
    "env_ids_for_task",
    "eval_spec_for_env",
    "register_pg3d_eval_envs",
    "task_spec_for_env",
]
