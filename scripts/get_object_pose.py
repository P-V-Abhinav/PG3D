#!/usr/bin/env python
"""
Utility script to get the ground-truth pose of the Banana 
in the PG3DReach-RealKitchen-v0 environment for a given dataset episode.

Usage:
    python scripts/get_banana_pose.py --dataset /scratch2/abhinav.pv/PG3D_artifacts/pg3d_reach_pose_multimodal_xarm7.zarr --episode 0
"""

import argparse
import zarr
import numpy as np
import gymnasium as gym
import json

from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.envs.xarm_adapter.obstacle_envs import *  # This registers the kitchen env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    # 1. Open the dataset and find the seed for this episode
    zarr_root = zarr.open_group(args.dataset, mode="r")
    meta_path = args.dataset.replace(".zarr", "") + "/metadata.json"
    
    try:
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        seed = int(metadata["episodes"][args.episode]["seed"])
        print(f"Loaded episode {args.episode} (Env Seed: {seed})")
    except Exception as e:
        print(f"Could not load seed from metadata.json, falling back to episode index as seed. Error: {e}")
        seed = args.episode

    # 2. Make the Kitchen environment
    # Note: We must use the RealKitchen env because that is the only one with the banana
    env = gym.make("PG3DReach-RealKitchen-v0", obs_mode="none")
    
    # 3. Reset the environment using the seed
    env.reset(seed=seed)
    
    # 4. Find the banana actor
    # In obstacle_envs.py, it's created in a loop where banana is index 8: "011_banana_8"
    banana_actor = None
    for actor in env.unwrapped.scene.get_all_actors():
        if "banana" in actor.name:
            banana_actor = actor
            break
            
    if banana_actor is None:
        print("❌ Could not find banana in the scene!")
        return
        
    # 5. Extract and print the pose
    pose = banana_actor.pose
    def to_1d_numpy(val):
        if hasattr(val, 'cpu'):
            val = val.cpu().numpy()
        return np.asarray(val).reshape(-1)
        
    pos = to_1d_numpy(pose.p)
    quat = to_1d_numpy(pose.q)
    
    print("\n🍌 Banana Ground-Truth Pose found!")
    print(f"  Position: X = {pos[0]:.4f}, Y = {pos[1]:.4f}, Z = {pos[2]:.4f}")
    
    # We want to approach from ABOVE the banana.
    # So we take the banana's X and Y, and add some height (e.g., +15cm) to Z.
    target_z = pos[2] + 0.15
    
    print("\nNext step: Run the IK script targeting a point 15cm above the banana:")
    print(f"python scripts/get_ik_joint_angles.py --x {pos[0]:.4f} --y {pos[1]:.4f} --z {target_z:.4f} --approach down")


if __name__ == "__main__":
    main()
