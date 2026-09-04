"""eval_envs/__init__.py — Register all 25 PG3D eval environments.

Importing this module triggers all @register_env decorators.
Call register_pg3d_eval_envs() from your eval script or add the
import to registration.py.
"""
from __future__ import annotations

_REGISTERED = False


def register_pg3d_eval_envs() -> None:
    """Lazily import all eval env modules to trigger @register_env decorators."""
    global _REGISTERED
    if _REGISTERED:
        return
    import pg3d.envs.xarm_adapter.eval_envs.reach_eval      # noqa: F401
    import pg3d.envs.xarm_adapter.eval_envs.pp_eval         # noqa: F401
    import pg3d.envs.xarm_adapter.eval_envs.obs_eval        # noqa: F401
    import pg3d.envs.xarm_adapter.eval_envs.cluttered_eval  # noqa: F401
    _REGISTERED = True
