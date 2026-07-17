import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# We import the rollout logic directly so we don't have to duplicate the setup
from rollout_dp3_reach_policy import (
    parse_args,
    load_reach_metadata,
    select_device,
    load_reach_policy_from_checkpoint,
    select_rollout_specs,
    crop_config_from_metadata,
    _action_mode,
    run_policy_rollout,
    register_pg3d_reach_envs,
    register_pg3d_xarm7_gripper_reach_envs,
    _policy_trajectory_family_count,
)

def main():
    parser = argparse.ArgumentParser(description="Plot multimodality by re-running policy in memory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, default=Path("artifacts/multimodality_plot.png"))
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--episode-index", type=int, default=0, help="The exact dataset seed to repeat.")
    args, unknown = parser.parse_known_args()

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(f"Failed to import ManiSkill: {exc}")
        return 1

    register_pg3d_reach_envs()
    register_pg3d_xarm7_gripper_reach_envs()
    device = select_device("auto")
    
    policy = load_reach_policy_from_checkpoint(args.checkpoint, device=device, prefer_ema=True)
    metadata = load_reach_metadata(args.dataset)
    
    # Repeat the EXACT SAME dataset episode seed N times
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    indices = [args.episode_index] * args.episodes
    specs = select_rollout_specs(
        source="dataset",
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=indices,
    )

    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["render_mode"] = None  # We don't need videos for this, making it lightning fast!
    env_kwargs.setdefault("obs_mode", "pointcloud")

    env = gym.make(str(metadata["env_id"]), **env_kwargs)
    
    all_trajectories = []
    start_pos = None
    goal_pos = None

    print(f"Running {args.episodes} rollouts on seed index {args.episode_index}...")
    
    trajectory_family_count = _policy_trajectory_family_count(policy)

    for i, spec in enumerate(specs):
        print(f"Evaluating trajectory {i+1}/{args.episodes}...")
        
        # We hook into the environment to manually track the TCP position
        obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
        
        tcp_path = []
        
        # We just run the environment step by step using the policy
        # Instead of calling run_policy_rollout (which creates rrd/mp4), we do a barebones loop
        from rollout_dp3_reach_policy import rollout_observation_entry, make_initial_obs_window, obs_window_to_torch, policy_action_to_sim_action, append_obs_window
        
        entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
        obs_window = make_initial_obs_window(entry, n_obs_steps=int(policy.n_obs_steps))
        
        if start_pos is None:
            start_pos = entry["tcp_pose"][:3]
            goal_pos = entry["target_position"]
            
        tcp_path.append(entry["tcp_pose"][:3])
        
        steps = 0
        success = False
        while steps < 150:
            import torch
            with torch.no_grad():
                batch = obs_window_to_torch(obs_window, device=device, goal_marker_points=int(policy.goal_marker_points), trajectory_family_count=trajectory_family_count)
                output = policy.predict_action(batch)
                action_chunk = output["action"][0].detach().cpu().numpy()
                
            for policy_action in action_chunk[:int(policy.n_action_steps)]: # replan stride
                sim_action = policy_action_to_sim_action(
                    policy_action, 
                    obs_window[-1]["agent_pos"], 
                    action_mode=action_mode, 
                    sim_action_dim=int(np.prod(env.action_space.shape))
                )
                obs, reward, terminated, truncated, info = env.step(sim_action)
                steps += 1
                
                entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
                obs_window = append_obs_window(obs_window, entry, n_obs_steps=int(policy.n_obs_steps))
                
                tcp_path.append(entry["tcp_pose"][:3])
                
                if info.get("success", False):
                    success = True
                    break
                    
            if success or terminated or truncated:
                break
                
        all_trajectories.append(np.array(tcp_path))
        print(f"  -> Finished in {steps} steps. Success: {success}")

    env.close()

    # --- PLOTTING ---
    print(f"\nPlotting all {len(all_trajectories)} trajectories...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(all_trajectories)))
    
    for idx, path in enumerate(all_trajectories):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=2.5, color=colors[idx], alpha=0.7, label=f"Path {idx+1}")
        
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='black', s=100, marker='x', label="Start")
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color='green', s=100, marker='o', label="Goal")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Multimodality Inference ({args.episodes} traces on exact same start/goal)")
    
    plt.tight_layout()
    
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_plot, dpi=300)
    print(f"\nSaved beautiful 3D multimodality plot to: {args.output_plot}")

if __name__ == "__main__":
    sys.exit(main())
