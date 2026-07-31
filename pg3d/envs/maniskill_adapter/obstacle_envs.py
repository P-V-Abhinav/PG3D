"""
obstacle_envs.py
================
Panda obstacle environment classes.
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

from pg3d.envs.maniskill_adapter.reach_env import PG3DReachEnv
from pg3d.envs.xarm_adapter.obstacle_envs import _create_cone_obj, _build_obstacle_cameras


@register_env("PG3DReach-Panda-RealMixedObstacle-v0", max_episode_steps=100)
class PG3DReachPandaRealMixedObstacleEnv(PG3DReachEnv):
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
            rng = np.random.default_rng(self._episode_seed)
            
            start_pos = self.agent.tcp_pose.p[0].cpu().numpy()
            goal_pos = self.goal_site.pose.p[0].cpu().numpy()
            mid_pos = (start_pos + goal_pos) / 2.0
            
            z_offsets = [0.15, 0.225, 0.04]
            
            def set_pose_batched(actor, pos_np):
                pos_t = torch.tensor(pos_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                actor.set_pose(Pose.create_from_pq(pos_t))
            
            vec = goal_pos[:2] - start_pos[:2]
            length = np.linalg.norm(vec)
            if length > 1e-4:
                perp = np.array([-vec[1], vec[0]]) / length
            else:
                perp = np.array([1.0, 0.0])
                
            spacing = 0.08
            
            p_box = mid_pos.copy()
            p_box[:2] += perp * spacing
            p_box[2] = z_offsets[0]
            set_pose_batched(self.obstacles[0], p_box)
            
            p_cone = mid_pos.copy()
            p_cone[:2] -= perp * spacing
            p_cone[2] = z_offsets[1]
            set_pose_batched(self.obstacles[1], p_cone)
            
            placed_positions = [start_pos[:2], goal_pos[:2], p_box[:2], p_cone[:2]]
            
            clutter_indices = [2]
            
            for idx in clutter_indices:
                placed = False
                for _ in range(100):
                    sx = rng.uniform(0.1, 0.6)
                    sy = rng.uniform(-0.4, 0.4)
                    cand = np.array([sx, sy])
                    
                    dists = [np.linalg.norm(cand - p) for p in placed_positions]
                    if dists[0] > 0.15 and dists[1] > 0.15 and all(d > 0.10 for d in dists[2:]):
                        pos = np.array([sx, sy, z_offsets[idx]])
                        set_pose_batched(self.obstacles[idx], pos)
                        placed_positions.append(cand)
                        placed = True
                        break
                
                if not placed:
                    set_pose_batched(self.obstacles[idx], np.array([0, 0, -1.0]))

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return _build_obstacle_cameras(super()._default_sensor_configs)
