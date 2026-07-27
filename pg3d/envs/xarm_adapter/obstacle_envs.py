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
        
        # 2. Cone
        obj_path = "/tmp/tall_cone.obj"
        _create_cone_obj(obj_path, radius=0.06, height=0.45)
        b_cone = self.scene.create_actor_builder()
        b_cone.add_convex_collision_from_file(obj_path)
        b_cone.add_visual_from_file(obj_path)
        self.cone_obs = b_cone.build_kinematic("obs_cone")
        
        # 3. Sphere
        self.sphere_obs = actors.build_sphere(
            self.scene, radius=0.04,
            color=[1.0, 0.5, 0.0, 1.0], name="obs_sphere", body_type="kinematic"
        )
        
        self.obstacles = [self.box_obs, self.cone_obs, self.sphere_obs]
        
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
            
            # Select which obstacle is the primary one in the middle
            obs_indices = [0, 1, 2]
            rng.shuffle(obs_indices)
            primary_idx = obs_indices[0]
            clutter_indices = obs_indices[1:]
            
            # Z offsets for centering based on object type
            # Box is 30cm tall, so center is 0.15. Cone is 45cm tall, center 0.225. Sphere r=4cm, center=0.04
            z_offsets = [0.15, 0.225, 0.04]
            
            # Place primary obstacle
            p_pos = mid_pos.copy()
            p_pos[2] = z_offsets[primary_idx]
            
            # To set poses in batched environments (tensor), we need (1, 3) arrays
            def set_pose_batched(actor, pos_np):
                pos_t = torch.tensor(pos_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                actor.set_pose(Pose.create_from_pq(pos_t))
                
            set_pose_batched(self.obstacles[primary_idx], p_pos)
            
            placed_positions = [start_pos[:2], goal_pos[:2], p_pos[:2]]
            
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
