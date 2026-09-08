"""xArm7 adapter: ManiSkill agents + reach envs for pg3d.

Sibling of :mod:`pg3d.envs.maniskill_adapter` (Panda). Importing either
register function imports the env module which registers all agents and env ids.

- :func:`register_pg3d_xarm7_reach_envs`        — no-gripper (TCP = link_eef)
- :func:`register_pg3d_xarm7_gripper_reach_envs` — with gripper (TCP = link_tcp)

Both share the same dataset schema, DP3 training, constraints, and eval.
"""

from __future__ import annotations


def register_pg3d_xarm7_reach_envs() -> None:
    """Register ``xarm7_nogripper`` agent and ``PG3DReach-XArm7-*`` env ids."""
    from pg3d.envs.xarm_adapter import reach_env  # noqa: F401


def register_pg3d_xarm7_gripper_reach_envs() -> None:
    """Register ``xarm7_gripper`` agent and ``PG3DReach-XArm7-Gripper-*`` env ids.

    Also registers the frozen eval suite (``PG3DReach-Eval-*``), which is built
    on this same agent. Every eval script already calls this registrar, so the
    suite's ids resolve in all of them without a per-script edit -- the point of
    the suite being self-contained is lost if using one still needs a code
    change. Registering ids no script asks for is inert.
    """
    from pg3d.envs.xarm_adapter import reach_env  # noqa: F401

    register_pg3d_eval_envs()


def register_pg3d_xarm7_robotiq_reach_envs() -> None:
    """Register ``xarm7_robotiq`` agent and ``PG3DReach-XArm7-Robotiq-*`` env ids."""
    from pg3d.envs.xarm_adapter import reach_env  # noqa: F401


def register_pg3d_eval_envs() -> None:
    """Register the frozen 25-env ten-task evaluation suite (``PG3DReach-Eval-*``).

    Local-testing mirror of the master suite; see
    :mod:`pg3d.envs.xarm_adapter.eval_envs.master_env_settings` for what this
    copy pins so it matches the official camera/arm layout.
    """
    from pg3d.envs.xarm_adapter.eval_envs import register_pg3d_eval_envs as _register

    _register()


__all__ = [
    "register_pg3d_eval_envs",
    "register_pg3d_xarm7_reach_envs",
    "register_pg3d_xarm7_gripper_reach_envs",
    "register_pg3d_xarm7_robotiq_reach_envs",
]
