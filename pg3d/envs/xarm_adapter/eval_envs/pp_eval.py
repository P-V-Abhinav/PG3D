"""pp_eval.py -- Deterministic pick-and-place eval envs (T3, T5, T7, T8).

Scene: the xArm7 + xArm parallel-jaw gripper, one dynamic 7 cm cube at a frozen
position, and the goal site pinned at the frozen place target. The arm starts at
``eval_config.START_TCP`` with the jaws OPEN so a pick phase can close them.

One env backs four tasks. The scene is identical; the task difference is the
constraint an eval script binds:

  T3 nominal pick and place      -- no extra constraint
  T5 pick from the left/right    -- approach-posture constraint on P1
  T7 pick at a given pose        -- pose goal + orientation constraint on P1
  T8 keep the object's pose      -- orientation constraint active through P3

Phase sequencing, gripper timing and grasp verification stay in the executor /
eval script: this module owns the scene and its frozen numbers only.
"""

from __future__ import annotations

import warnings
from typing import Any

import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .eval_base import PG3DEvalBase
from .eval_config import (
    CUBE_HALF_SIZE,
    CUBE_POSITIONS,
    PLACE_TARGETS,
    PP_START_QPOS,
    PP_START_TCPS,
)
from .master_env_settings import GRIPPER_CONTROL_MODE

#: Re-exported for the obstacle and cluttered modules. The value comes from the
#: repo seam because the two checkouts name it differently: in the master repo
#: the commandable-gripper mode is ``pd_joint_pos_gripper`` (plain
#: ``pd_joint_pos`` pins the jaws with a static hold controller), while the older
#: repo's gripper agent already exposes the mimic controller under
#: ``pd_joint_pos``.
__all__ = ["GRIPPER_CONTROL_MODE", "PG3DEvalPickPlaceEnv"]


class PG3DEvalPickPlaceEnv(PG3DEvalBase):
    """Pick a frozen cube and place it at a frozen target."""

    TASK_IDS = ("T3", "T5", "T7", "T8")

    #: World-frame resting centre of the cube.
    CUBE_POS: tuple[float, float, float]

    CUBE_HALF_SIZE: float = CUBE_HALF_SIZE
    GRIPPER_START = "open"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The env, not the caller, owns the control mode a pick needs. A caller
        # that passes something else (e.g. a reach dataset's recorded control
        # mode, whose jaws cannot be commanded) would get an ungraspable
        # episode, so the override is applied and announced rather than
        # silently honoured.
        requested = kwargs.get("control_mode")
        if requested not in (None, GRIPPER_CONTROL_MODE):
            warnings.warn(
                f"[{type(self).__name__}] overriding control_mode={requested!r} with "
                f"{GRIPPER_CONTROL_MODE!r}: the gripper must be commandable for a pick task.",
                stacklevel=2,
            )
        kwargs["control_mode"] = GRIPPER_CONTROL_MODE
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        # Friction and contact patch match the M4 pick-and-place cube, so a
        # closed jaw holds the cube instead of squeezing it out.
        material = sapien.physx.PhysxMaterial(
            static_friction=2.0, dynamic_friction=2.0, restitution=0.0
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=material,
            patch_radius=0.05,
            min_patch_radius=0.05,
        )
        builder.add_box_visual(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=sapien.render.RenderMaterial(base_color=[0.85, 0.20, 0.20, 1.0]),
        )
        builder.set_initial_pose(sapien.Pose(p=list(self.CUBE_POS)))
        self.cube = builder.build(name="cube")
        # Canonical handle every task-agnostic consumer should read.
        self.pg3d_grasp_target = self.cube
        # Compatibility alias: eval_graspgen_pick.py resolves the object to grasp
        # by looking for `env.unwrapped.cheezit` (the actor name in the kitchen /
        # jstbanana envs it was written against) and its grasp-signal helper
        # reads that attribute directly. Aliasing keeps that script working
        # against this suite with no edit; drop it once the script reads
        # `pg3d_grasp_target`.
        self.cheezit = self.pg3d_grasp_target

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        batch = len(env_idx)
        with torch.device(self.device):
            cube = torch.tensor(
                [list(self.CUBE_POS)], dtype=torch.float32, device=self.device
            ).expand(batch, -1)
            self.cube.set_pose(Pose.create_from_pq(cube))
            zeros = torch.zeros((batch, 3), dtype=torch.float32, device=self.device)
            self.cube.set_linear_velocity(zeros)
            self.cube.set_angular_velocity(zeros)

    def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        extra = super()._get_obs_extra(info)
        extra["obj_pose"] = self.cube.pose.raw_pose
        return extra

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Reach grading from the parent, plus placement/grasp diagnostics.

        ``success`` is left as the parent's TCP-to-goal predicate; terminal
        placement success belongs to the phase executor, which owns the
        stability hold (D-41).
        """
        result = super().evaluate()
        result["is_obj_placed"] = (
            torch.linalg.norm(self.cube.pose.p - self.goal_site.pose.p, axis=1) <= 0.05
        )
        result["is_grasped"] = self.agent.is_grasping(self.cube)
        return result


@register_env("PG3DReach-Eval-PP-v1", max_episode_steps=200)
class PG3DEvalPPV1(PG3DEvalPickPlaceEnv):
    """PP v1 -- front-centre cube, lateral transport to the far front right."""

    CUBE_POS = CUBE_POSITIONS["v1"]
    GOAL_POS = PLACE_TARGETS["v1"]
    START_TCP_POS = PP_START_TCPS["v1"]
    START_QPOS = PP_START_QPOS["v1"]


@register_env("PG3DReach-Eval-PP-v2", max_episode_steps=200)
class PG3DEvalPPV2(PG3DEvalPickPlaceEnv):
    """PP v2 -- far-forward cube, longitudinal pull-back to the back left."""

    CUBE_POS = CUBE_POSITIONS["v2"]
    GOAL_POS = PLACE_TARGETS["v2"]
    START_TCP_POS = PP_START_TCPS["v2"]
    START_QPOS = PP_START_QPOS["v2"]


@register_env("PG3DReach-Eval-PP-v3", max_episode_steps=200)
class PG3DEvalPPV3(PG3DEvalPickPlaceEnv):
    """PP v3 -- far front-right cube, diagonal cross-sweep to the back left."""

    CUBE_POS = CUBE_POSITIONS["v3"]
    GOAL_POS = PLACE_TARGETS["v3"]
    START_TCP_POS = PP_START_TCPS["v3"]
    START_QPOS = PP_START_QPOS["v3"]


@register_env("PG3DReach-Eval-PP-v4", max_episode_steps=200)
class PG3DEvalPPV4(PG3DEvalPickPlaceEnv):
    """PP v4 -- far front-left cube, diagonal pull-back to the back right."""

    CUBE_POS = CUBE_POSITIONS["v4"]
    GOAL_POS = PLACE_TARGETS["v4"]
    START_TCP_POS = PP_START_TCPS["v4"]
    START_QPOS = PP_START_QPOS["v4"]


@register_env("PG3DReach-Eval-PP-v5", max_episode_steps=200)
class PG3DEvalPPV5(PG3DEvalPickPlaceEnv):
    """PP v5 -- close right-offset cube, diagonal sweep to the far front left."""

    CUBE_POS = CUBE_POSITIONS["v5"]
    GOAL_POS = PLACE_TARGETS["v5"]
    START_TCP_POS = PP_START_TCPS["v5"]
    START_QPOS = PP_START_QPOS["v5"]
