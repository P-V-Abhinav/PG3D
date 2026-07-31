#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import zarr

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a joint configuration from a Zarr dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the Zarr dataset.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to extract from.")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction along the trajectory to sample (0.0 to 1.0).")
    args = parser.parse_args()

    try:
        root = zarr.open_group(str(args.dataset), mode="r")
    except Exception as exc:
        print(f"Failed to open dataset {args.dataset}: {exc}")
        return 1
    
    if "meta" not in root or "episode_ends" not in root["meta"]:
        print("Dataset missing meta/episode_ends array.")
        return 1
        
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if not (0 <= args.episode < len(episode_ends)):
        print(f"Episode {args.episode} is out of bounds (dataset has {len(episode_ends)} episodes).")
        return 1
        
    start = 0 if args.episode == 0 else int(episode_ends[args.episode - 1])
    end = int(episode_ends[args.episode])
    horizon = end - start
    
    if horizon <= 0:
        print(f"Episode {args.episode} is empty.")
        return 1
        
    data = root["data"]
    # Check possible joint state array keys
    joint_array = None
    key_used = None
    for key in ["state", "qpos", "sim_action", "action"]:
        if key in data:
            joint_array = data[key]
            key_used = key
            break
            
    if joint_array is None:
        print(f"Could not find joint data (state, qpos, sim_action, action) in data/")
        return 1

    target_offset = int(round(args.fraction * (horizon - 1)))
    abs_idx = start + target_offset
    joint_data = np.asarray(joint_array[abs_idx])
    
    # XArm7 reach tasks use 7 joints for the arm
    target_joints = joint_data[:7]
    
    print(f"Extracted joints from episode {args.episode} (slice {start}:{end}, offset {target_offset}/{horizon-1}, key '{key_used}'):")
    print(" ".join(f"{j:.6f}" for j in target_joints))
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
