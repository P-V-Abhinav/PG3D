"""reach_eval.py — Deterministic Reach evaluation environments (Task 1).

5 variants, hardcoded goal positions, no randomness.

Inherits from PG3DReachXArm7GripperEnv which provides:
  - xarm7_gripper robot at ROBOT_BASE_POSE = [-0.615, 0, 0]
  - Camera jitter + calibration error (kept — matches real-world deployment)
  - start_site (red sphere) and goal_site (green sphere) actors
  - compute_dense_reward / evaluate() from PG3DReachEnv

New behaviour added here:
  - goal_site / start_site rendered strictly VIRTUAL (no point cloud)
  - goal_site placed at the hardcoded world-frame position for this variant
  - Optional green workspace bounding-box wireframe (--show-workspace)
  - Assertion: start_site within START_SITE_REST_TOLERANCE_M of rest TCP
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from pg3d.envs.xarm_adapter.reach_env import PG3DReachXArm7GripperEnv

from .eval_config import (
    REACH_GOALS,
    START_SITE_REST_TOLERANCE_M,
    workspace_box_edges,
)


# ---------------------------------------------------------------------------
# Utility: strip render component → actor becomes strictly virtual
# ---------------------------------------------------------------------------

def _set_actor_visibility(actor: Any, visibility: float) -> None:
    """Set visibility of an actor. Used to hide spheres during point cloud 
    generation but keep them visible to the human viewer.
    """
    try:
        import sapien.render as sr
    except ImportError:
        return

    for obj in getattr(actor, "_objs", [actor]):
        body = obj.find_component_by_type(sr.RenderBodyComponent)
        if body is not None:
            if hasattr(body, "set_visibility"):
                body.set_visibility(visibility)
            elif hasattr(body, "set_visible"):
                body.set_visible(visibility > 0.5)



# ---------------------------------------------------------------------------
# Base class for ALL eval environments
# ---------------------------------------------------------------------------

class PG3DEvalBase(PG3DReachXArm7GripperEnv):
    """Shared base for all PG3D eval environments.

    Responsibilities
    ----------------
    1. Make start_site and goal_site strictly virtual in _load_scene.
    2. Build the workspace bounding-box wireframe if show_workspace=True.
    3. Check start-site distance from rest TCP each episode.
    """

    def __init__(
        self,
        *args: Any,
        show_workspace: bool = False,
        **kwargs: Any,
    ) -> None:
        self._show_workspace = show_workspace
        self._workspace_actors: list[Any] = []
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)

        # ── 2. Optional workspace wireframe ───────────────────────────────
        if self._show_workspace:
            self._workspace_actors = []
            for edge in workspace_box_edges():
                cx, cy, cz = edge["centre"]
                bar = actors.build_box(
                    self.scene,
                    half_sizes=list(edge["half_sizes"]),
                    color=[0.0, 1.0, 0.0, 0.8],
                    name=f"ws_edge_{len(self._workspace_actors)}",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[cx, cy, cz]),
                )
                self._workspace_actors.append(bar)

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        super()._initialize_episode(env_idx, options)
        # ── 3. Check start-site proximity to rest TCP ─────────────────────
        with torch.no_grad():
            start_pos = self.agent.tcp_pose.p  # (N, 3)
            site_pos = self.start_site.pose.p  # (N, 3)
            dist = torch.linalg.norm(site_pos - start_pos, dim=-1)
            max_dist = float(dist.max().item())
        if max_dist > START_SITE_REST_TOLERANCE_M:
            warnings.warn(
                f"[{self.__class__.__name__}] start_site is {max_dist:.4f} m "
                f"from rest TCP (tolerance {START_SITE_REST_TOLERANCE_M} m). "
                "Check that _initialize_episode sets start_site = tcp at rest.",
                stacklevel=2,
            )

    def get_obs(self, info: dict[str, Any] | None = None, unflattened: bool = False) -> Any:
        # Hide markers right before generating observations (so they don't appear in point cloud)
        _set_actor_visibility(self.goal_site, 0.0)
        if hasattr(self, "start_site"):
            _set_actor_visibility(self.start_site, 0.0)
        
        obs = super().get_obs(info, unflattened=unflattened)
        
        # Restore markers so they remain visible in the human viewer
        _set_actor_visibility(self.goal_site, 1.0)
        if hasattr(self, "start_site"):
            _set_actor_visibility(self.start_site, 1.0)
            
        return obs


# ---------------------------------------------------------------------------
# ENV 1: Reach  (Task 1)
# ---------------------------------------------------------------------------

class PG3DEvalReachEnv(PG3DEvalBase):
    """Deterministic reach environment base.

    Subclasses set GOAL_POS = (x, y, z) in world frame.
    _initialize_episode places goal_site at that fixed position.
    """

    GOAL_POS: tuple[float, float, float]  # subclass must define

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


@register_env("PG3DReach-Eval-Reach-v1", max_episode_steps=150)
class PG3DEvalReachV1(PG3DEvalReachEnv):
    """Reach v1 — front-center nominal.  Goal: (-0.280, 0.000, 0.200)."""
    GOAL_POS = REACH_GOALS["v1"]


@register_env("PG3DReach-Eval-Reach-v2", max_episode_steps=150)
class PG3DEvalReachV2(PG3DEvalReachEnv):
    """Reach v2 — far-forward, near table.  Goal: (-0.165, 0.000, 0.080)."""
    GOAL_POS = REACH_GOALS["v2"]


@register_env("PG3DReach-Eval-Reach-v3", max_episode_steps=150)
class PG3DEvalReachV3(PG3DEvalReachEnv):
    """Reach v3 — full left lateral.  Goal: (-0.300, 0.380, 0.220)."""
    GOAL_POS = REACH_GOALS["v3"]


@register_env("PG3DReach-Eval-Reach-v4", max_episode_steps=150)
class PG3DEvalReachV4(PG3DEvalReachEnv):
    """Reach v4 — full right lateral.  Goal: (-0.300, -0.380, 0.220)."""
    GOAL_POS = REACH_GOALS["v4"]


@register_env("PG3DReach-Eval-Reach-v5", max_episode_steps=150)
class PG3DEvalReachV5(PG3DEvalReachEnv):
    """Reach v5 — high upward.  Goal: (-0.370, 0.000, 0.350)."""
    GOAL_POS = REACH_GOALS["v5"]
