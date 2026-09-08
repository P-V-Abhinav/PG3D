"""cluttered_eval.py -- Deterministic cluttered pick-and-place eval envs (T9, T10).

Five frozen YCB layouts: one target object plus 3-8 clutter objects, all at
frozen positions and yaws, with the goal site at the frozen place target. The
same envs back both tasks:

  T9  clutter, every non-target object treated as an obstacle
  T10 relational placement (left / right / on top of a reference object)

T10 needs no scene change: the eval script computes the place goal from a
reference object's live pose and binds the relation as a constraint, then calls
:meth:`PG3DEvalClutteredEnv.set_place_goal`.

Assets
------
Objects are built through ManiSkill's own YCB builder
(``actors.get_actor_builder(scene, id="ycb:<model>")``), which is the same path
the rest of this repo uses, so the meshes, scales and collision decomposition
are ManiSkill's. There is no fallback shape: a missing YCB asset raises, because
a box standing in for a mug silently changes the task.

Resting height
--------------
The frozen layouts carry a hand-estimated z per object. YCB local origins are
NOT centred and differ per model, so a hand-estimated z leaves some objects
floating and others interpenetrating the table (which makes the first physics
step explode). By default the z is replaced with the exact resting height
derived from the same ``info_pick_v0.json`` metadata ManiSkill builds the actor
from; the frozen x/y and yaw are untouched. Pass ``use_ycb_rest_z=False`` to use
the frozen z verbatim.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation as _Rotation

from .eval_base import PG3DEvalBase
from .eval_config import (
    CLUTTERED_LAYOUTS,
    CLUTTERED_START_QPOS,
    CLUTTERED_START_TCPS,
)
from .pp_eval import GRIPPER_CONTROL_MODE

Vec3 = tuple[float, float, float]


def _ycb_rest_z(model_id: str) -> float | None:
    """World z at which ``model_id``'s underside just touches the table.

    Reads the bbox metadata ManiSkill itself uses when building the actor, so
    the number matches the mesh that is actually loaded. TableSceneBuilder puts
    the table's top surface at world z = 0, which is the datum here. Returns
    None when the model is absent from the metadata.
    """
    try:
        import mani_skill
        from mani_skill.utils.io_utils import load_json

        info = load_json(mani_skill.ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        meta = info[model_id]
        scale = meta.get("scales", [1.0])[0]
        return float(-meta["bbox"]["min"][2] * scale)
    except Exception as exc:
        warnings.warn(
            f"could not read the YCB resting height for {model_id!r} "
            f"({type(exc).__name__}: {exc}); using the frozen z.",
            stacklevel=2,
        )
        return None


def _yaw_quat_wxyz(yaw_deg: float) -> list[float]:
    quat_xyzw = _Rotation.from_euler("z", float(np.deg2rad(yaw_deg))).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


class PG3DEvalClutteredEnv(PG3DEvalBase):
    """Pick a frozen YCB target out of frozen clutter and place it."""

    TASK_IDS = ("T9", "T10")

    #: ``"v1"``..``"v5"``, selecting a layout from ``CLUTTERED_LAYOUTS``.
    LAYOUT_KEY: str

    GRIPPER_START = "open"

    def __init__(self, *args: Any, use_ycb_rest_z: bool = True, **kwargs: Any) -> None:
        self._use_ycb_rest_z = bool(use_ycb_rest_z)
        requested = kwargs.get("control_mode")
        if requested not in (None, GRIPPER_CONTROL_MODE):
            warnings.warn(
                f"[{type(self).__name__}] overriding control_mode={requested!r} with "
                f"{GRIPPER_CONTROL_MODE!r}: the gripper must be commandable for a pick task.",
                stacklevel=2,
            )
        kwargs["control_mode"] = GRIPPER_CONTROL_MODE
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    @classmethod
    def layout(cls) -> dict:
        return CLUTTERED_LAYOUTS[cls.LAYOUT_KEY]

    def _resting_position(self, model_id: str, position: Vec3) -> Vec3:
        if not self._use_ycb_rest_z:
            return position
        rest_z = _ycb_rest_z(model_id)
        if rest_z is None:
            return position
        return (position[0], position[1], rest_z)

    def object_positions(self) -> dict[str, Vec3]:
        """Frozen world positions actually used, keyed by actor name."""
        layout = self.layout()
        target = layout["target"]
        result = {"target_object": self._resting_position(target["model"], target["pos"])}
        for index, item in enumerate(layout["clutter"]):
            result[f"clutter_{index}"] = self._resting_position(item["model"], item["pos"])
        return result

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _build_ycb(self, model_id: str, name: str) -> Any:
        try:
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        except Exception as exc:
            raise RuntimeError(
                f"YCB model {model_id!r} could not be built for {type(self).__name__}. "
                "Download the ManiSkill YCB assets "
                "(`python -m mani_skill.utils.download_asset ycb`) before running the "
                "cluttered eval envs; this suite has no substitute shapes."
            ) from exc
        builder.set_scene_idxs(None)
        return builder.build(name=name)

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        layout = self.layout()
        self.target_object = self._build_ycb(layout["target"]["model"], "target_object")
        self.clutter_objects: list[Any] = [
            self._build_ycb(item["model"], f"clutter_{index}")
            for index, item in enumerate(layout["clutter"])
        ]
        # Canonical handle every task-agnostic consumer should read.
        self.pg3d_grasp_target = self.target_object
        # Compatibility alias: eval_graspgen_pick.py resolves the object to grasp
        # by looking for `env.unwrapped.cheezit` (the actor name in the kitchen /
        # jstbanana envs it was written against) and its grasp-signal helper
        # reads that attribute directly. Aliasing keeps that script working
        # against this suite with no edit; drop it once the script reads
        # `pg3d_grasp_target`.
        self.cheezit = self.pg3d_grasp_target

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        layout = self.layout()
        positions = self.object_positions()
        batch = len(env_idx)
        with torch.device(self.device):
            self._place_object(
                self.target_object, positions["target_object"], layout["target"]["yaw_deg"], batch
            )
            for index, (actor, item) in enumerate(
                zip(self.clutter_objects, layout["clutter"], strict=True)
            ):
                self._place_object(actor, positions[f"clutter_{index}"], item["yaw_deg"], batch)

    def _place_object(self, actor: Any, position: Vec3, yaw_deg: float, batch: int) -> None:
        pose = torch.tensor(
            [list(position)], dtype=torch.float32, device=self.device
        ).expand(batch, -1)
        quat = torch.tensor(
            [_yaw_quat_wxyz(yaw_deg)], dtype=torch.float32, device=self.device
        ).expand(batch, -1)
        actor.set_pose(Pose.create_from_pq(pose, quat))
        zeros = torch.zeros((batch, 3), dtype=torch.float32, device=self.device)
        if hasattr(actor, "set_linear_velocity"):
            actor.set_linear_velocity(zeros)
            actor.set_angular_velocity(zeros)

    def set_place_goal(self, position: Any) -> None:
        """Move the graded goal (T10's relational target) at eval time.

        T9 never calls this: its place goal is the frozen one. T10 resolves a
        relation against a reference object's live pose and calls this once,
        before the transport phase.
        """
        pose = torch.as_tensor(position, dtype=torch.float32, device=self.device).reshape(1, 3)
        batch = self.goal_site.pose.p.shape[0]
        expanded = pose.expand(batch, -1)
        self.goal_site.set_pose(Pose.create_from_pq(expanded))

    def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        extra = super()._get_obs_extra(info)
        extra["obj_pose"] = self.target_object.pose.raw_pose
        return extra

    def evaluate(self) -> dict[str, torch.Tensor]:
        result = super().evaluate()
        result["is_obj_placed"] = (
            torch.linalg.norm(self.target_object.pose.p - self.goal_site.pose.p, axis=1) <= 0.05
        )
        result["is_grasped"] = self.agent.is_grasping(self.target_object)
        return result


@register_env("PG3DReach-Eval-Cluttered-v1", max_episode_steps=300)
class PG3DEvalClutteredV1(PG3DEvalClutteredEnv):
    """Cluttered v1 -- loose layout, mug target, 3 clutter objects."""

    LAYOUT_KEY = "v1"
    GOAL_POS = CLUTTERED_LAYOUTS["v1"]["place_goal"]
    START_TCP_POS = CLUTTERED_START_TCPS["v1"]
    START_QPOS = CLUTTERED_START_QPOS["v1"]


@register_env("PG3DReach-Eval-Cluttered-v2", max_episode_steps=300)
class PG3DEvalClutteredV2(PG3DEvalClutteredEnv):
    """Cluttered v2 -- dense cluster, mustard-bottle target, 4 clutter objects."""

    LAYOUT_KEY = "v2"
    GOAL_POS = CLUTTERED_LAYOUTS["v2"]["place_goal"]
    START_TCP_POS = CLUTTERED_START_TCPS["v2"]
    START_QPOS = CLUTTERED_START_QPOS["v2"]


@register_env("PG3DReach-Eval-Cluttered-v3", max_episode_steps=300)
class PG3DEvalClutteredV3(PG3DEvalClutteredEnv):
    """Cluttered v3 -- arc of clutter, banana target, 5 clutter objects."""

    LAYOUT_KEY = "v3"
    GOAL_POS = CLUTTERED_LAYOUTS["v3"]["place_goal"]
    START_TCP_POS = CLUTTERED_START_TCPS["v3"]
    START_QPOS = CLUTTERED_START_QPOS["v3"]


@register_env("PG3DReach-Eval-Cluttered-v4", max_episode_steps=300)
class PG3DEvalClutteredV4(PG3DEvalClutteredEnv):
    """Cluttered v4 -- mixed sizes blocking the path, soup-can target."""

    LAYOUT_KEY = "v4"
    GOAL_POS = CLUTTERED_LAYOUTS["v4"]["place_goal"]
    START_TCP_POS = CLUTTERED_START_TCPS["v4"]
    START_QPOS = CLUTTERED_START_QPOS["v4"]


@register_env("PG3DReach-Eval-Cluttered-v5", max_episode_steps=300)
class PG3DEvalClutteredV5(PG3DEvalClutteredEnv):
    """Cluttered v5 -- maximum 8-object clutter, gelatin-box target."""

    LAYOUT_KEY = "v5"
    GOAL_POS = CLUTTERED_LAYOUTS["v5"]["place_goal"]
    START_TCP_POS = CLUTTERED_START_TCPS["v5"]
    START_QPOS = CLUTTERED_START_QPOS["v5"]
