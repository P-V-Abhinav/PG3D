"""
obstacle_envs.py
================
xArm7 obstacle environment classes for the real-obstacle reach evaluation.

Extracted from scripts/eval_pointcloud_obstacle_reach.py so that both the
eval script AND Ray parallel workers can import them without either one
having to import the full heavyweight eval script.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from pg3d.envs.xarm_adapter.reach_env import PG3DReachXArm7GripperEnv


def _hide_marker_spheres(env: Any) -> None:
    """Hide the start_site and goal_site markers so they do not show in point clouds."""
    try:
        import sapien.render as sr
    except ImportError:
        return
        
    for attr in ("start_site", "goal_site"):
        actor = getattr(env, attr, None)
        if actor is not None:
            for obj in getattr(actor, "_objs", [actor]):
                body = obj.find_component_by_type(sr.RenderBodyComponent)
                if body is not None:
                    body.set_visible(False)


def _create_cone_obj(
    filepath: str,
    radius: float = 0.05,
    height: float = 0.40,
    segments: int = 32,
) -> str:
    """Generate a simple 3D cone OBJ file and return its path."""
    if os.path.exists(filepath):
        return filepath

    with open(filepath, "w") as f:
        # Top vertex
        f.write(f"v 0 0 {height / 2}\n")
        # Bottom center
        f.write(f"v 0 0 {-height / 2}\n")
        # Bottom ring
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            f.write(f"v {x} {y} {-height / 2}\n")

        # Top (lateral) faces
        for i in range(segments):
            curr = i + 3
            next_idx = 3 if i == segments - 1 else i + 4
            f.write(f"f 1 {curr} {next_idx}\n")

        # Bottom cap faces
        for i in range(segments):
            curr = i + 3
            next_idx = 3 if i == segments - 1 else i + 4
            f.write(f"f 2 {next_idx} {curr}\n")

    return filepath


# ---------------------------------------------------------------------------
# Shared camera configuration used by both obstacle env classes
# ---------------------------------------------------------------------------
_OBSTACLE_CAM_CONFIGS = {
    "cam_front_left": {
        "eye":    [-0.5,  1.65, 0.85],
        "target": [-0.5, -0.0,  0.40],
        "fov_deg": 60.0,
    },
    "cam_side_right": {
        "eye":    [-0.5, -1.65, 0.85],
        "target": [-0.5, -0.05, 0.40],
        "fov_deg": 60.0,
    },
    "cam_overhead": {
        "eye":    [0.20, 0.00, 1.20],
        "target": [-0.30, 0.00, 0.40],
        "fov_deg": 70.0,
    },
    "cam_back": {
        "eye":    [-1.50, 0.00, 0.85],
        "target": [-0.50, 0.00, 0.40],
        "fov_deg": 60.0,
    },
}


def _build_obstacle_cameras(base_configs: list[CameraConfig]) -> list[CameraConfig]:
    """Append the 4 obstacle cameras to an existing sensor config list."""
    configs = list(base_configs)
    for name, cfg in _OBSTACLE_CAM_CONFIGS.items():
        pose = sapien_utils.look_at(eye=cfg["eye"], target=cfg["target"])
        configs.append(
            CameraConfig(
                name,
                pose,
                128,
                128,
                float(np.deg2rad(cfg["fov_deg"])),
                0.1,
                10.0,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# xArm7 box obstacle environment
# ---------------------------------------------------------------------------
@register_env("PG3DReach-XArm7-RealObstacle-v0", max_episode_steps=100)
class PG3DReachXArm7RealObstacleEnv(PG3DReachXArm7GripperEnv):
    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        self.obstacle = actors.build_box(
            self.scene,
            half_sizes=[0.03, 0.03, 0.15],
            color=[0.0, 0.0, 1.0, 1.0],
            name="obstacle",
            body_type="kinematic",
        )
        _hide_marker_spheres(self)

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            start_pos = self.agent.tcp_pose.p
            goal_pos = self.goal_site.pose.p
            mid_pos = (start_pos + goal_pos) / 2.0
            mid_pos[:, 2] = 0.15
            self.obstacle.set_pose(Pose.create_from_pq(mid_pos))

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return _build_obstacle_cameras(super()._default_sensor_configs)


# ---------------------------------------------------------------------------
# xArm7 tall-cone obstacle environment
# ---------------------------------------------------------------------------
@register_env("PG3DReach-RealConeObstacle-v0", max_episode_steps=100)
class PG3DReachRealConeObstacleEnv(PG3DReachXArm7GripperEnv):
    def _load_scene(self, options: dict[str, Any] | None) -> None:
        super()._load_scene(options)
        obj_path = "/tmp/tall_cone.obj"
        _create_cone_obj(obj_path, radius=0.06, height=0.45)
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(obj_path)
        builder.add_visual_from_file(obj_path)
        self.obstacle = builder.build_kinematic("obstacle")
        _hide_marker_spheres(self)

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            start_pos = self.agent.tcp_pose.p
            goal_pos = self.goal_site.pose.p
            mid_pos = (start_pos + goal_pos) / 2.0
            mid_pos[:, 2] = 0.225  # centre of a 0.45 m tall cone
            self.obstacle.set_pose(Pose.create_from_pq(mid_pos))

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return _build_obstacle_cameras(super()._default_sensor_configs)


# ---------------------------------------------------------------------------
# Mixed Obstacles (Cone, Box, Sphere) with Random Scattering
# ---------------------------------------------------------------------------
@register_env("PG3DReach-RealMixedObstacle-v0", max_episode_steps=100)
class PG3DReachRealMixedObstacleEnv(PG3DReachXArm7GripperEnv):
    def _load_scene(self, options: dict[str, Any] | None) -> None:
        super()._load_scene(options)
        
        # 1. Box
        self.box_obs = actors.build_box(
            self.scene, half_sizes=[0.03, 0.03, 0.15],
            color=[0.0, 0.0, 1.0, 1.0], name="obs_box", body_type="kinematic"
        )
        
        # 2. Box 2 (replaces cone)
        self.box_obs_2 = actors.build_box(
            self.scene, half_sizes=[0.03, 0.03, 0.15],
            color=[0.0, 1.0, 0.0, 1.0], name="obs_box_2", body_type="kinematic"
        )

        # 3. Box 3 (midpoint, 2cm ahead)
        self.box_obs_3 = actors.build_box(
            self.scene, half_sizes=[0.03, 0.03, 0.15],
            color=[1.0, 0.0, 0.0, 1.0], name="obs_box_3", body_type="kinematic"
        )
        
        # 4. Sphere
        self.sphere_obs = actors.build_sphere(
            self.scene, radius=0.04,
            color=[1.0, 0.5, 0.0, 1.0], name="obs_sphere", body_type="kinematic"
        )
        
        self.obstacles = [self.box_obs, self.box_obs_2, self.box_obs_3, self.sphere_obs]
        _hide_marker_spheres(self)
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            # We want deterministic placement per episode based on np.random which is seeded
            # The base env seeds np.random in `reset(seed=...)` or `_initialize_episode`.
            # We will use torch random state or np.random. 
            # In mani_skill, `self._episode_seed` is available to seed RNGs deterministically.
            rng = np.random.default_rng(self._episode_seed)
            
            start_pos = self.agent.tcp_pose.p[0].cpu().numpy()
            goal_pos = self.goal_site.pose.p[0].cpu().numpy()
            mid_pos = (start_pos + goal_pos) / 2.0
            
            # Z offsets for centering based on object type
            z_offsets = [0.15, 0.15, 0.15, 0.04]
            
            # To set poses in batched environments (tensor), we need (1, 3) arrays
            def set_pose_batched(actor, pos_np):
                pos_t = torch.tensor(pos_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                actor.set_pose(Pose.create_from_pq(pos_t))
            
            # Place both Box 1 (0) and Box 2 (1) in the middle, spaced apart perpendicularly
            vec = goal_pos[:2] - start_pos[:2]
            length = np.linalg.norm(vec)
            if length > 1e-4:
                perp = np.array([-vec[1], vec[0]]) / length
                forward = vec / length
            else:
                perp = np.array([1.0, 0.0])
                forward = np.array([0.0, 1.0])
                
            spacing = 0.08  # 8cm offset from center for each
            
            p_box1 = mid_pos.copy()
            p_box1[:2] += perp * spacing
            p_box1[2] = z_offsets[0]
            set_pose_batched(self.obstacles[0], p_box1)
            
            p_box2 = mid_pos.copy()
            p_box2[:2] -= perp * spacing
            p_box2[2] = z_offsets[1]
            set_pose_batched(self.obstacles[1], p_box2)
            
            # Midpoint, 2cm ahead along the path
            p_box3 = mid_pos.copy()
            p_box3[:2] += forward * 0.02
            p_box3[2] = z_offsets[2]
            set_pose_batched(self.obstacles[2], p_box3)
            
            placed_positions = [start_pos[:2], goal_pos[:2], p_box1[:2], p_box2[:2], p_box3[:2]]
            
            # Scatter Sphere (3)
            clutter_indices = [3]
            
            # Place clutter
            for idx in clutter_indices:
                placed = False
                for _ in range(100):
                    # Table roughly spans X: [-0.2, 0.2], Y: [-0.4, 0.4] around base.
                    # Base is at origin. Let's sample X in [0.1, 0.6], Y in [-0.4, 0.4] to be in front.
                    sx = rng.uniform(0.1, 0.6)
                    sy = rng.uniform(-0.4, 0.4)
                    cand = np.array([sx, sy])
                    
                    # Check distances
                    dists = [np.linalg.norm(cand - p) for p in placed_positions]
                    # Must be at least 15cm from start/goal, and 10cm from other obstacles
                    if dists[0] > 0.15 and dists[1] > 0.15 and all(d > 0.10 for d in dists[2:]):
                        pos = np.array([sx, sy, z_offsets[idx]])
                        set_pose_batched(self.obstacles[idx], pos)
                        placed_positions.append(cand)
                        placed = True
                        break
                
                # Fallback if we couldn't place it safely (e.g., hide it under table)
                if not placed:
                    set_pose_batched(self.obstacles[idx], np.array([0, 0, -1.0]))

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return _build_obstacle_cameras(super()._default_sensor_configs)


# ---------------------------------------------------------------------------
# Cluttered Kitchen Environment (YCB Objects)
# ---------------------------------------------------------------------------
@register_env("PG3DReach-RealKitchen-v0", max_episode_steps=100)
class PG3DReachRealKitchenEnv(PG3DReachXArm7GripperEnv):
    def _load_scene(self, options: dict[str, Any] | None) -> None:
        super()._load_scene(options)
        
        self.ycb_objects = []
        # Added 5 more complex/large YCB objects to heavily increase OOD clutter
        model_ids = [
            "025_mug", "024_bowl", "005_tomato_soup_can", "006_mustard_bottle", "009_gelatin_box",
            "003_cracker_box", "004_sugar_box", "010_potted_meat_can", "011_banana", "021_bleach_cleanser"
        ]
        
        for i, model_id in enumerate(model_ids):
            try:
                # ManiSkill 3 standard asset loading
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.set_scene_idxs(None) # applies to all scenes
                actor = builder.build_kinematic(name=f"{model_id}_{i}")
                self.ycb_objects.append(actor)
            except Exception as e:
                print(f"[PG3DReachRealKitchenEnv] Failed to load YCB {model_id}: {e}")
                actor = actors.build_box(self.scene, half_sizes=[0.04, 0.04, 0.1], color=[1, 0, 0, 1], name=f"fallback_{i}", body_type="kinematic")
                self.ycb_objects.append(actor)
        _hide_marker_spheres(self)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            rng = np.random.default_rng(self._episode_seed)
            
            start_pos = self.agent.tcp_pose.p[0].cpu().numpy()
            goal_pos = self.goal_site.pose.p[0].cpu().numpy()
            
            # Lower the goal position so it is amidst the clutter but not touching the ground
            goal_pos[2] = 0.08  # 8cm above the table
            goal_pos_t = torch.tensor(goal_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_pos_t))
            
            def set_pose_batched(actor, pos_np, q_np):
                pos_t = torch.tensor(pos_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_t = torch.tensor(q_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                actor.set_pose(Pose.create_from_pq(p=pos_t, q=q_t))
            
            placed_positions = [start_pos[:2], goal_pos[:2]]
            
            for i, actor in enumerate(self.ycb_objects):
                placed = False
                for _ in range(500):
                    # Obstacles placed closely around the goal to obstruct the direct path
                    # Expanded max radius to 35cm to fit all 10 objects
                    angle = rng.uniform(0, 2 * np.pi)
                    rad = rng.uniform(0.08, 0.35) 
                    sx = goal_pos[0] + rad * np.cos(angle)
                    sy = goal_pos[1] + rad * np.sin(angle)
                    cand = np.array([sx, sy])
                    
                    dists = [np.linalg.norm(cand - p) for p in placed_positions]
                    # Ensure it doesn't overlap the start (12cm), is very close to goal but not overlapping (8cm), and doesn't overlap other clutter (9cm)
                    if dists[0] > 0.12 and dists[1] > 0.08 and all(d > 0.09 for d in dists[2:]):
                        theta = rng.uniform(0, 2 * np.pi)
                        q = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
                        
                        pos = np.array([sx, sy, 0.05])
                        set_pose_batched(actor, pos, q)
                        placed_positions.append(cand)
                        placed = True
                        break
                
                if not placed:
                    set_pose_batched(actor, np.array([0, 0, -1.0]), [1, 0, 0, 0])

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return _build_obstacle_cameras(super()._default_sensor_configs)


# ---------------------------------------------------------------------------
# Just Banana Environment (Now Cheezit Box)
# ---------------------------------------------------------------------------
@register_env("jstbanana-v0", max_episode_steps=100)
class PG3DReachJustBananaEnv(PG3DReachXArm7GripperEnv):
    def _load_scene(self, options: dict[str, Any] | None) -> None:
        super()._load_scene(options)
        model_id = "003_cracker_box"
        try:
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.set_scene_idxs(None)
            self.cheezit = builder.build_kinematic(name=f"{model_id}")
        except Exception as e:
            print(f"[PG3DReachJustBananaEnv] Failed to load YCB {model_id}: {e}")
            self.cheezit = actors.build_box(self.scene, half_sizes=[0.02, 0.08, 0.02], color=[1, 1, 0, 1], name="fallback_cheezit", body_type="kinematic")
        _hide_marker_spheres(self)

    # Workspace XY bounds for random object placement in jstbanana-v0.
    # Tighter bounds to ensure the object spawns comfortably within robot reach.
    _JSTBANANA_X_RANGE = (-0.15, 0.07)
    _JSTBANANA_Y_RANGE = (-0.25, 0.07)
    _JSTBANANA_Z     = 0.08  # table surface height for object base
    _JSTBANANA_MIN_DIST_FROM_START = 0.20   # keep object away from robot home

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            rng = np.random.default_rng(self._episode_seed)
            start_pos = self.agent.tcp_pose.p[0].cpu().numpy()[:2]  # XY of robot TCP

            # --- Random placement within workspace crop bounds ---
            placed = False
            for _ in range(200):
                sx = rng.uniform(*self._JSTBANANA_X_RANGE)
                sy = rng.uniform(*self._JSTBANANA_Y_RANGE)
                if np.linalg.norm(np.array([sx, sy]) - start_pos) >= self._JSTBANANA_MIN_DIST_FROM_START:
                    placed = True
                    break
            if not placed:
                # Fallback: use centre of workspace
                sx, sy = 0.30, 0.0

            pos_np = np.array([sx, sy, self._JSTBANANA_Z], dtype=np.float32)
            pos_t = torch.tensor(pos_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Random rotation around Z axis
            theta = rng.uniform(0, 2 * np.pi)
            q_t = torch.tensor(
                [np.cos(theta / 2), 0, 0, np.sin(theta / 2)],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0)

            self.cheezit.set_pose(Pose.create_from_pq(p=pos_t, q=q_t))
            # Move the goal marker to exactly the cheezit centroid so the
            # reach-policy goal marker (blue sphere) coincides with the object.
            self.goal_site.set_pose(Pose.create_from_pq(p=pos_t))

            print(
                f"[jstbanana-v0] Episode seed={self._episode_seed}: "
                f"cheezit placed at ({sx:.3f}, {sy:.3f}, {self._JSTBANANA_Z:.3f})",
                flush=True,
            )

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return _build_obstacle_cameras(super()._default_sensor_configs)
