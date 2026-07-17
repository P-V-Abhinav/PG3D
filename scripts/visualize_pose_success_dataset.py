#!/usr/bin/env python
"""Visualize EEF pose success rates and plot TCP trajectories from a Zarr dataset."""

import argparse
import json
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/pg3d_reach_pose_multimodal.zarr"))
    parser.add_argument("--output", type=Path, default=Path("pose_success_viz.png"))
    parser.add_argument("--max-episodes", type=int, default=1000, help="Max episodes to plot to avoid clutter")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset {args.dataset} not found.")
        return 1

    meta_path = args.dataset / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    episodes = meta.get("episodes", [])
    if not episodes:
        print("No episode metadata found.")
        return 1

    print(f"Loaded metadata for {len(episodes)} episodes.")

    # Group by orientation mode
    stats = {}
    for i, ep in enumerate(episodes):
        mode = ep.get("orientation_mode", "unknown")
        success = ep.get("success", False)
        if mode not in stats:
            stats[mode] = {"total": 0, "success": 0, "indices": []}
        stats[mode]["total"] += 1
        if success:
            stats[mode]["success"] += 1
            stats[mode]["indices"].append(i)

    print("\n--- Success Rates by Orientation Mode ---")
    for mode, data in stats.items():
        rate = (data["success"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"{mode.ljust(20)}: {rate:5.1f}% ({data['success']}/{data['total']})")
    print("-----------------------------------------\n")

    print("Loading TCP trajectories from Zarr...")
    z = zarr.open(str(args.dataset), mode="r")
    tcp_poses = z["data"]["tcp_pose"][:]
    episode_ends = z["meta"]["episode_ends"][:]

    fig = plt.figure(figsize=(15, 5))
    modes = list(stats.keys())
    
    for plot_idx, mode in enumerate(modes):
        ax = fig.add_subplot(1, len(modes), plot_idx + 1, projection='3d')
        ax.set_title(f"{mode} (Successes)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Plot up to args.max_episodes successful trajectories for this mode
        indices_to_plot = stats[mode]["indices"][:args.max_episodes]
        for ep_idx in indices_to_plot:
            start_step = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            end_step = episode_ends[ep_idx]
            
            traj = tcp_poses[start_step:end_step, :3] # Only XYZ
            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3)
                # Mark goal
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=10)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to {args.output.absolute()}")

if __name__ == "__main__":
    raise SystemExit(main())
