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
    
    metadata_json = root.attrs.get("metadata")
    if metadata_json:
        metadata = json.loads(metadata_json)
        episodes = metadata.get("episodes", [])
        if not (0 <= args.episode < len(episodes)):
            print(f"Episode {args.episode} is out of bounds (dataset has {len(episodes)} episodes).")
            return 1
    
    episode_group = root.get(f"episode_{args.episode:06d}")
    if episode_group is None:
        print(f"Could not find episode_{args.episode:06d} in dataset.")
        return 1

    try:
        qpos = np.asarray(episode_group["data"]["qpos"])
    except KeyError:
        print("Could not find data/qpos in the episode.")
        return 1

    horizon = qpos.shape[0]
    if horizon == 0:
        print("Trajectory is empty.")
        return 1
    
    target_idx = int(round(args.fraction * (horizon - 1)))
    # XArm7 reach tasks typically use the first 7 joints for the arm
    target_joints = qpos[target_idx, :7]
    
    print(f"Extracted joints from episode {args.episode} at timestep {target_idx}/{horizon-1} (fraction {args.fraction}):")
    # Output the joints as a space-separated string suitable for CLI arguments
    print(" ".join(f"{j:.6f}" for j in target_joints))
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
