"""cluttered_eval.py — Deterministic Cluttered Pick & Place environments (Tasks 9, 10).

5 variants with fixed YCB object layouts.

Scene contents
--------------
  XArm7 + xarm7_gripper
  Target YCB object  (self.target_object)
  N clutter YCB objects  (self.clutter_objects)
  goal_site (strictly virtual) at place target
  start_site (strictly virtual) at rest TCP

Task 10 (Place Relative to Object) reuses these exact envs.
The relative placement goal is computed in the eval wrapper as:
  place_goal = reference_object.pose.p + offset_world
No env change is needed.

No fallback objects are used. If a YCB model fails to load the env
raises an error — this is intentional and ensures layouts are exact.

YCB loading
-----------
Uses the same sapien URDF/mesh loading approach as PG3DReachRealKitchenEnv.
The YCB dataset must be available at the standard ManiSkill YCB asset path.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import sapien
import torch
import numpy as np
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation as _Rotation

from .eval_config import CLUTTERED_LAYOUTS
from .reach_eval import PG3DEvalBase


# ---------------------------------------------------------------------------
# YCB asset loading helper
# ---------------------------------------------------------------------------

def _get_ycb_asset_path() -> Path:
    """Return the ManiSkill YCB asset directory."""
    import mani_skill
    ms_root = Path(mani_skill.__file__).parent
    # Standard ManiSkill YCB path
    ycb_path = ms_root / "assets" / "mani_skill2_ycb" / "models"
    if not ycb_path.exists():
        # Alternate path used in some installations
        ycb_path = ms_root / "data" / "mani_skill2_ycb" / "models"
    if not ycb_path.exists():
        raise FileNotFoundError(
            f"YCB asset directory not found. Tried:\n"
            f"  {ms_root / 'assets' / 'mani_skill2_ycb' / 'models'}\n"
            f"  {ms_root / 'data'   / 'mani_skill2_ycb' / 'models'}\n"
            "Ensure ManiSkill YCB assets are downloaded."
        )
    return ycb_path


def _load_ycb_object(
    scene: Any,
    model_id: str,
    name: str,
    *,
    ycb_path: Path,
) -> Any:
    """Load a YCB object into the ManiSkill scene.

    Attempts to load via sapien articulation builder from the YCB URDF.
    Raises RuntimeError if the model file is not found — no fallback.
    """
    model_dir = ycb_path / model_id
    urdf_path = model_dir / "model.urdf"
    if not urdf_path.exists():
        # Try textured_simple.obj → build a convex mesh actor
        obj_path = model_dir / "textured_simple.obj"
        if not obj_path.exists():
            raise RuntimeError(
                f"YCB model '{model_id}' not found at {model_dir}. "
                "Download YCB assets before running cluttered envs."
            )
    # Use ManiSkill's actor builder with convex decomposition
    builder = scene.create_actor_builder()
    builder.set_scene_idxs(scene.actor_builder._scene_idxs if hasattr(scene, 'actor_builder') else [0])
    if urdf_path.exists():
        # Load from URDF (preserves visual mesh)
        loader = scene.get_urdf_loader()
        loader.name = name
        loader.load(str(urdf_path))
        # Return the last actor added (URDF loader registers it)
        return list(scene.actors.values())[-1]
    else:
        # Convex mesh fallback
        builder.add_convex_collision_from_file(str(obj_path))
        builder.add_visual_from_file(str(obj_path))
        builder.set_name(name)
        return builder.build_kinematic()


# ---------------------------------------------------------------------------
# Cluttered env base
# ---------------------------------------------------------------------------

class PG3DEvalClutteredEnv(PG3DEvalBase):
    """Deterministic cluttered Pick & Place base.

    Subclasses set LAYOUT_KEY = "v1" .. "v5" to select the layout
    from CLUTTERED_LAYOUTS in eval_config.py.
    """

    LAYOUT_KEY: str  # subclass must define; one of "v1".."v5"

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        layout = CLUTTERED_LAYOUTS[self.LAYOUT_KEY]
        ycb_path = _get_ycb_asset_path()

        # Load target object
        tgt = layout["target"]
        self.target_object = _load_ycb_object(
            self.scene,
            tgt["model"],
            name="target_object",
            ycb_path=ycb_path,
        )

        # Load clutter objects
        self.clutter_objects: list[Any] = []
        for i, item in enumerate(layout["clutter"]):
            obj = _load_ycb_object(
                self.scene,
                item["model"],
                name=f"clutter_{i}",
                ycb_path=ycb_path,
            )
            self.clutter_objects.append(obj)

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        super()._initialize_episode(env_idx, options)
        layout = CLUTTERED_LAYOUTS[self.LAYOUT_KEY]

        with torch.device(self.device):
            n = len(env_idx)

            # Place target object
            tgt = layout["target"]
            pos_t = torch.tensor([list(tgt["pos"])] * n, dtype=torch.float32, device=self.device)
            yaw_rad = float(np.deg2rad(tgt["yaw_deg"]))
            q_xyzw = _Rotation.from_euler("z", yaw_rad).as_quat()   # xyzw
            q_wxyz = torch.tensor(
                [[q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]] * n,
                dtype=torch.float32,
                device=self.device,
            )
            self.target_object.set_pose(Pose.create_from_pq(p=pos_t, q=q_wxyz))

            # Place clutter objects
            for obj, item in zip(self.clutter_objects, layout["clutter"]):
                pos_c = torch.tensor([list(item["pos"])] * n, dtype=torch.float32, device=self.device)
                yaw_c = float(np.deg2rad(item["yaw_deg"]))
                q_c_xyzw = _Rotation.from_euler("z", yaw_c).as_quat()
                q_c = torch.tensor(
                    [[q_c_xyzw[3], q_c_xyzw[0], q_c_xyzw[1], q_c_xyzw[2]]] * n,
                    dtype=torch.float32,
                    device=self.device,
                )
                obj.set_pose(Pose.create_from_pq(p=pos_c, q=q_c))

            # Place goal_site at the place target
            place_t = torch.tensor(
                [list(layout["place_goal"])] * n,
                dtype=torch.float32,
                device=self.device,
            )
            self.goal_site.set_pose(Pose.create_from_pq(place_t))


# ---------------------------------------------------------------------------
# Cluttered variants  v1 – v5
# (Tasks 9 and 10 both use these registered envs)
# ---------------------------------------------------------------------------

@register_env("PG3DReach-Eval-Cluttered-v1", max_episode_steps=300)
class PG3DEvalClutteredV1(PG3DEvalClutteredEnv):
    """Cluttered v1 — loose layout, 3 clutter objects, mug target."""
    LAYOUT_KEY = "v1"


@register_env("PG3DReach-Eval-Cluttered-v2", max_episode_steps=300)
class PG3DEvalClutteredV2(PG3DEvalClutteredEnv):
    """Cluttered v2 — dense cluster, 4 clutter objects, mustard bottle target."""
    LAYOUT_KEY = "v2"


@register_env("PG3DReach-Eval-Cluttered-v3", max_episode_steps=300)
class PG3DEvalClutteredV3(PG3DEvalClutteredEnv):
    """Cluttered v3 — arc of 5 clutter objects, banana target."""
    LAYOUT_KEY = "v3"


@register_env("PG3DReach-Eval-Cluttered-v4", max_episode_steps=300)
class PG3DEvalClutteredV4(PG3DEvalClutteredEnv):
    """Cluttered v4 — mixed sizes blocking the path, soup-can target."""
    LAYOUT_KEY = "v4"


@register_env("PG3DReach-Eval-Cluttered-v5", max_episode_steps=300)
class PG3DEvalClutteredV5(PG3DEvalClutteredEnv):
    """Cluttered v5 — maximum 8-object clutter, gelatin box target."""
    LAYOUT_KEY = "v5"
