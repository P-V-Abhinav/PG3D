"""pp_eval.py — Deterministic Pick & Place evaluation environments (Tasks 2, 5, 7, 8).

5 variants sharing canonical CUBE_POSITIONS / PLACE_TARGETS from eval_config.

Scene contents
--------------
  XArm7 + xarm7_gripper
  dynamic 7 cm red cube
  goal_site  (strictly virtual green sphere) — at place target
  start_site (strictly virtual red sphere)   — at rest TCP

The environment does NOT implement gripper closing or scoring beyond
the inherited PG3DReachEnv.evaluate() / compute_dense_reward().  Gripper
close logic (ramp 0→0.85 over 30 steps, triggered at dist≤0.02m or step≥150)
lives in the eval script's post_episode_callback — matching the architecture
of eval_graspgen_pick.py.

Tasks reusing these envs (no env change needed, handled in eval wrapper):
  Task 5 — Pick from L/R   : ApproachPostureConstraint(left/right) at eval time
  Task 7 — Pick at Pose     : ApproachPostureConstraint per grasp preset
  Task 8 — Keep Pose        : orientation deviation tracked in eval wrapper
"""
from __future__ import annotations

import warnings
from typing import Any

import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation as _Rotation

from .eval_config import CUBE_POSITIONS, PLACE_TARGETS
from .reach_eval import PG3DEvalBase


# ---------------------------------------------------------------------------
# Pick & Place base
# ---------------------------------------------------------------------------

class PG3DEvalPPEnv(PG3DEvalBase):
    """Deterministic Pick & Place base.

    Subclasses set:
      CUBE_POS   : (x, y, z) — cube resting position on table (z ≈ 0.035)
      PLACE_POS  : (x, y, z) — where the cube should be placed
    """

    CUBE_POS:  tuple[float, float, float]   # subclass must define
    PLACE_POS: tuple[float, float, float]   # subclass must define

    # Cube geometry
    CUBE_HALF_SIZE: float = 0.035   # 7 cm cube

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        # Dynamic cube — no collision needed beyond physics; add_collision=True
        # so ManiSkill physics can detect contact (used by eval wrapper).
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

            # ── Place cube ────────────────────────────────────────────────
            cube_t = torch.tensor(
                [list(self.CUBE_POS)] * n,
                dtype=torch.float32,
                device=self.device,
            )
            self.cube.set_pose(Pose.create_from_pq(cube_t))

            # ── Place goal_site at place target ───────────────────────────
            place_t = torch.tensor(
                [list(self.PLACE_POS)] * n,
                dtype=torch.float32,
                device=self.device,
            )
            self.goal_site.set_pose(Pose.create_from_pq(place_t))


# ---------------------------------------------------------------------------
# Pick & Place variants  v1 – v5
# (Tasks 2, 5, 7, 8 all reuse these exact registered envs)
# ---------------------------------------------------------------------------

@register_env("PG3DReach-Eval-PP-v1", max_episode_steps=200)
class PG3DEvalPPV1(PG3DEvalPPEnv):
    """PP v1 — front-center cube, lateral transport.
    Cube: (-0.280,  0.000, 0.035)  Place: (-0.280,  0.250, 0.035)"""
    CUBE_POS  = CUBE_POSITIONS["v1"]
    PLACE_POS = PLACE_TARGETS["v1"]


@register_env("PG3DReach-Eval-PP-v2", max_episode_steps=200)
class PG3DEvalPPV2(PG3DEvalPPEnv):
    """PP v2 — far-forward cube, longitudinal pull-back.
    Cube: (-0.170,  0.000, 0.035)  Place: (-0.400,  0.000, 0.035)"""
    CUBE_POS  = CUBE_POSITIONS["v2"]
    PLACE_POS = PLACE_TARGETS["v2"]


@register_env("PG3DReach-Eval-PP-v3", max_episode_steps=200)
class PG3DEvalPPV3(PG3DEvalPPEnv):
    """PP v3 — right-offset cube, cross-sweep R→L.
    Cube: (-0.300, -0.180, 0.035)  Place: (-0.300,  0.200, 0.035)"""
    CUBE_POS  = CUBE_POSITIONS["v3"]
    PLACE_POS = PLACE_TARGETS["v3"]


@register_env("PG3DReach-Eval-PP-v4", max_episode_steps=200)
class PG3DEvalPPV4(PG3DEvalPPEnv):
    """PP v4 — left-offset cube, diagonal transport.
    Cube: (-0.250,  0.150, 0.035)  Place: (-0.170, -0.150, 0.035)"""
    CUBE_POS  = CUBE_POSITIONS["v4"]
    PLACE_POS = PLACE_TARGETS["v4"]


@register_env("PG3DReach-Eval-PP-v5", max_episode_steps=200)
class PG3DEvalPPV5(PG3DEvalPPEnv):
    """PP v5 — center cube, vertical lift.
    Cube: (-0.310,  0.000, 0.035)  Place: (-0.310,  0.000, 0.220)"""
    CUBE_POS  = CUBE_POSITIONS["v5"]
    PLACE_POS = PLACE_TARGETS["v5"]
