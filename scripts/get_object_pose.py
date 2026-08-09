"""
Utility script to get the ground-truth pose of an object 
in the PG3DReach-RealKitchen-v0 environment for a given dataset episode.

Usage:
    python scripts/get_object_pose.py --dataset /scratch2/abhinav.pv/PG3D_artifacts/pg3d_reach_pose_multimodal_xarm7.zarr --episode 0 --object banana
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
    parser.add_argument("--object", type=str, default="banana", help="The name of the object to find (e.g., 'banana', 'mug', 'bowl', 'mustard')")
    args = parser.parse_args()

    # 1. Open the dataset and find the seed for this episode
    # metadata.json lives INSIDE the zarr directory
    import os
    meta_path = os.path.join(args.dataset, "metadata.json")
    
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
    
    # 4. Find the target object actor
    target_actor = None
    target_obj_name = args.object.lower()
    for actor in env.unwrapped.scene.get_all_actors():
        if target_obj_name in actor.name.lower():
            target_actor = actor
            break
            
    if target_actor is None:
        print(f"❌ Could not find '{args.object}' in the scene!")
        print("Available objects:")
        for actor in env.unwrapped.scene.get_all_actors():
            if actor.name.startswith("0"):
                print(f"  - {actor.name}")
        return
        
    # 5. Extract and print the pose
    pose = target_actor.pose
    def to_1d_numpy(val):
        if hasattr(val, 'cpu'):
            val = val.cpu().numpy()
        return np.asarray(val).reshape(-1)
        
    pos = to_1d_numpy(pose.p)
    quat = to_1d_numpy(pose.q)
    
    print(f"\n✅ {args.object.capitalize()} Ground-Truth Pose found!")
    print(f"  Position: X = {pos[0]:.4f}, Y = {pos[1]:.4f}, Z = {pos[2]:.4f}")
    
    # We want to approach from ABOVE the object.
    target_z = pos[2] + 0.15
    
    print(f"\nNext step: Run the IK script targeting a point 15cm above the {args.object}:")
    print(f"python scripts/get_ik_joint_angles.py --x {pos[0]:.4f} --y {pos[1]:.4f} --z {target_z:.4f} --approach downward")


if __name__ == "__main__":
    main()
