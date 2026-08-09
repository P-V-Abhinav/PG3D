#!/usr/bin/env python
"""
Utility script to convert a Cartesian End-Effector Pose (XYZ + Quaternion)
into a 7-DOF Joint Angle Configuration using Inverse Kinematics (IK).

CRITICAL: Pass --dataset and --episode so the IK solver is seeded from the
episode's actual start joint configuration. This ensures the returned target
joints are kinematically reachable from the arm's true starting state for
that episode, rather than an arbitrary rest pose.

Usage Example:
    python scripts/get_ik_joint_angles.py \\
        --dataset /scratch2/abhinav.pv/PG3D_artifacts/pg3d_reach_pose_multimodal_xarm7.zarr \\
        --episode 1499 \\
        --x 0.0673 --y 0.2257 --z 0.2257 \\
        --approach pitch_45
"""

import argparse
import numpy as np
import zarr
import gymnasium as gym
import mani_skill.envs  # noqa: F401

from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver


def _load_episode_start_qpos(dataset_path: str, episode_index: int) -> np.ndarray:
    """Load the arm's starting joint configuration for a specific episode from zarr."""
    zarr_root = zarr.open_group(dataset_path, mode="r")
    episode_ends = np.asarray(zarr_root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_index < 0 or episode_index >= len(episode_ends):
        raise IndexError(
            f"Episode index {episode_index} is out of range [0, {len(episode_ends) - 1}]"
        )
    episode_start_row = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    # state stores the full joint qpos at each timestep; row 0 is the episode start
    start_qpos = np.asarray(zarr_root["data"]["state"][episode_start_row], dtype=np.float32)
    return start_qpos.flatten()


def get_ik(x, y, z, qw, qx, qy, qz, *, start_qpos_seed: np.ndarray | None = None):
    # 1. Initialize the environment to get the robot kinematic model
    register_pg3d_xarm7_gripper_reach_envs()
    env = gym.make(
        "PG3DReach-XArm7-Gripper-Workspace-v0",
        obs_mode="none",
        render_mode="rgb_array",
    )
    env.reset(seed=0)

    # 2. Build the motion planner (IK solver)
    solver = XArm7GripperMotionPlanningSolver(
        env, debug=False, vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
    )
    planner = solver.planner  # The raw mplib planner

    # 3. Format the goal pose for the IK solver
    # mplib expects a 7D vector: [x, y, z, qw, qx, qy, qz] in world frame
    goal_pose = np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)

    # 4. Build the IK seed with the correct full dimension.
    #
    # mplib's IK() requires the FULL robot qpos (all joints, including gripper
    # fingers). For XArm7GripperMotionPlanningSolver this is 13 dims (7 arm + 6
    # gripper finger joints). The zarr dataset only stores the 7 arm joints in
    # `state`. So we:
    #   1. Start from the robot's current full qpos (after env.reset), which
    #      gives a valid 13-dim baseline (arm at rest, gripper fully open).
    #   2. Overwrite just the first 7 values with the episode's arm start qpos.
    # This seeds the IK solver from the arm's actual starting configuration,
    # while keeping the gripper joints at their default (open) values.
    full_default_qpos = (
        np.asarray(env.unwrapped.agent.robot.get_qpos())
        .reshape(-1)
        .astype(np.float64)
    )
    if start_qpos_seed is not None:
        seed = full_default_qpos.copy()
        arm_vals = np.asarray(start_qpos_seed, dtype=np.float64).flatten()
        n_arm = min(len(arm_vals), len(seed))
        seed[:n_arm] = arm_vals[:n_arm]
        print(f"  IK Seed:     Episode start qpos ({n_arm} arm joints) + default gripper")
    else:
        seed = full_default_qpos
        print("  IK Seed:     Robot full rest pose (no --dataset/--episode provided!)")
        print("  ⚠️  WARNING: This may give target joints that don't match your episode.")


    # 5. Solve IK
    print(f"\nSolving IK for Cartesian Pose:")
    print(f"  Position:    X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
    print(f"  Orientation: qw={qw:.4f}, qx={qx:.4f}, qy={qy:.4f}, qz={qz:.4f}")

    status, plan = planner.IK(goal_pose, seed, n_init_qpos=20, threshold=1e-3)

    if status == "Success":
        # plan might have extra dimensions (e.g., shape [1, 7]), so flatten to 1D
        joint_angles = np.array(plan, dtype=np.float32).flatten()
        print("\n✅ IK Solution Found!")
        print("Use the following 7 values for your --posture-target-joints argument:\n")
        formatted_joints = " ".join([f"{angle:.6f}" for angle in joint_angles])
        print(f"  {formatted_joints}\n")
        print(f"Python list: {joint_angles.tolist()}")
    else:
        print("❌ IK Failed! The requested Cartesian pose is unreachable from this seed.")
        print(f"   Planner status: {status}")
        print("   Try adjusting --z (e.g., higher above the object) or --approach angle.")

    solver.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Cartesian pose to Joint Angles via IK, seeded from episode start."
    )

    # Target Cartesian position
    parser.add_argument("--x", type=float, required=True, help="Target X position (meters)")
    parser.add_argument("--y", type=float, required=True, help="Target Y position (meters)")
    parser.add_argument("--z", type=float, required=True, help="Target Z position (meters)")

    # Episode seed source (STRONGLY recommended)
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to zarr dataset. Required to seed IK from episode start qpos."
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Episode index in the dataset. Required to seed IK from episode start qpos."
    )

    # Approach direction (must match training data orientations exactly)
    parser.add_argument(
        "--approach", type=str,
        choices=["downward", "pitch_30", "pitch_45", "pitch_60", "horizontal_front", "custom"],
        default="downward",
        help=(
            "Gripper approach orientation. These EXACTLY match the training data orientations:\n"
            "  downward        = gripper straight down (default)\n"
            "  pitch_30        = 30 degrees up from vertical\n"
            "  pitch_45        = 45 degrees up from vertical\n"
            "  pitch_60        = 60 degrees up from vertical\n"
            "  horizontal_front = fully horizontal (90 degrees)\n"
            "  custom          = use --qw/--qx/--qy/--qz"
        ),
    )

    # Manual quaternions only if approach == 'custom'
    parser.add_argument("--qw", type=float, default=0.0, help="Quaternion W (only for --approach custom)")
    parser.add_argument("--qx", type=float, default=1.0, help="Quaternion X (only for --approach custom)")
    parser.add_argument("--qy", type=float, default=0.0, help="Quaternion Y (only for --approach custom)")
    parser.add_argument("--qz", type=float, default=0.0, help="Quaternion Z (only for --approach custom)")

    args = parser.parse_args()

    # Map approach name → exact quaternion from the training dataset
    APPROACH_QUATERNIONS = {
        "downward":         (0.0000, 1.0000, 0.0000,  0.0000),
        "pitch_30":         (0.0000, 0.9659, 0.0000, -0.2588),
        "pitch_45":         (0.0000, 0.9239, 0.0000, -0.3827),
        "pitch_60":         (0.0000, 0.8660, 0.0000, -0.5000),
        "horizontal_front": (0.0000, 0.7071, 0.0000, -0.7071),
    }

    if args.approach == "custom":
        qw, qx, qy, qz = args.qw, args.qx, args.qy, args.qz
    else:
        qw, qx, qy, qz = APPROACH_QUATERNIONS[args.approach]

    # Load episode start qpos as IK seed if dataset/episode provided
    start_qpos_seed = None
    if args.dataset is not None and args.episode is not None:
        try:
            start_qpos_seed = _load_episode_start_qpos(args.dataset, args.episode)
            print(f"Loaded episode {args.episode} start qpos from dataset as IK seed.")
            print(f"  Start qpos ({len(start_qpos_seed)} joints): {start_qpos_seed.tolist()}")
        except Exception as e:
            print(f"⚠️  Could not load episode start qpos: {e}")
            print("   Falling back to robot rest pose as seed.")
    elif args.dataset is None or args.episode is None:
        print("⚠️  --dataset and --episode not both provided.")
        print("   Falling back to robot rest pose as IK seed.")
        print("   For accurate posture steering, re-run with --dataset and --episode.\n")

    get_ik(args.x, args.y, args.z, qw, qx, qy, qz, start_qpos_seed=start_qpos_seed)
