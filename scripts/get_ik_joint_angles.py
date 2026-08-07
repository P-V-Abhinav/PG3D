#!/usr/bin/env python
"""
Utility script to convert a Cartesian End-Effector Pose (XYZ + Quaternion)
into a 7-DOF Joint Angle Configuration using Inverse Kinematics (IK).

You can use the output of this script directly as the argument for
--posture-target-joints in your pose steering evaluation script.

Usage Example:
    python scripts/get_ik_joint_angles.py --x 0.3 --y 0.0 --z 0.2 --qw 0.0 --qx 1.0 --qy 0.0 --qz 0.0
"""

import argparse
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: F401

from pg3d.envs.xarm_adapter import register_pg3d_xarm7_gripper_reach_envs
from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver

def get_ik(x, y, z, qw, qx, qy, qz):
    # 1. Initialize the environment just to get the robot kinematic model
    register_pg3d_xarm7_gripper_reach_envs()
    env = gym.make(
        "PG3DReach-XArm7-Gripper-Workspace-v0", 
        obs_mode="none", 
        render_mode="rgb_array"
    )
    env.reset(seed=0)
    
    # 2. Build the motion planner (IK solver)
    solver = XArm7GripperMotionPlanningSolver(
        env, debug=False, vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
    )
    planner = solver.planner # The raw mplib planner
    
    # 3. Format the goal pose for the IK solver
    # mplib expects a 7D vector: [x, y, z, qw, qx, qy, qz]
    # The positions must be in the world frame. We assume the robot base is at the origin [0,0,0]
    goal_pose = np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)
    
    # The IK solver needs an initial guess for the joint angles. 
    # We use the robot's default rest pose as the starting guess.
    n_ik = len(planner.user_joint_names)
    start_qpos = np.asarray(env.unwrapped.agent.robot.get_qpos()).reshape(-1)[:n_ik].astype(np.float64)
    
    # 4. Solve IK
    print(f"Solving IK for Cartesian Pose:")
    print(f"  Position:    X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    print(f"  Orientation: qw={qw:.3f}, qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f}\n")
    
    status, plan = planner.IK(goal_pose, start_qpos, n_init_qpos=20, threshold=1e-3)
    
    if status == "Success":
        # plan is a list/array of joint angles
        # plan might have extra dimensions (e.g., shape [1, 7]), so we flatten it to a 1D array of floats
        joint_angles = np.array(plan, dtype=np.float32).flatten()
        print("✅ IK Solution Found!")
        print("Use the following 7 values for your --posture-target-joints argument:\n")
        
        # Format it exactly as required by argparse (space-separated string)
        formatted_joints = " ".join([f"{angle:.6f}" for angle in joint_angles])
        print(f"  {formatted_joints}\n")
        
        # Also print as a Python list just in case
        print(f"Python list: {joint_angles.tolist()}")
    else:
        print("❌ IK Failed! The requested Cartesian pose is unreachable by the arm.")
        print(f"Planner status: {status}")
        
    solver.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Cartesian pose to Joint Angles via IK")
    parser.add_argument("--x", type=float, required=True, help="Target X position (meters)")
    parser.add_argument("--y", type=float, required=True, help="Target Y position (meters)")
    parser.add_argument("--z", type=float, required=True, help="Target Z position (meters)")
    
    # Approach direction helper
    parser.add_argument("--approach", type=str, choices=["down", "left", "right", "front", "back", "custom"], default="down", 
                        help="Pre-defined approach directions (gripper orientation). 'down' = approach from above.")
    
    # Manual quaternions if approach == 'custom'
    parser.add_argument("--qw", type=float, default=0.0, help="Quaternion W (used if --approach custom)")
    parser.add_argument("--qx", type=float, default=1.0, help="Quaternion X (used if --approach custom)")
    parser.add_argument("--qy", type=float, default=0.0, help="Quaternion Y (used if --approach custom)")
    parser.add_argument("--qz", type=float, default=0.0, help="Quaternion Z (used if --approach custom)")
    
    args = parser.parse_args()
    
    # Define exact quaternions for SAPIEN xArm7 TCP
    # Base frame: Z is UP, X is FORWARD, Y is LEFT
    if args.approach == "down":
        # Gripper points straight down (-Z)
        qw, qx, qy, qz = 0.0, 1.0, 0.0, 0.0
    elif args.approach == "front":
        # Gripper points inward from the front (+X direction)
        qw, qx, qy, qz = 0.5, 0.5, 0.5, 0.5
    elif args.approach == "back":
        # Gripper points outward from the back (-X direction)
        qw, qx, qy, qz = 0.5, -0.5, -0.5, 0.5
    elif args.approach == "left":
        # Gripper points inward from the left (-Y direction)
        qw, qx, qy, qz = 0.7071068, 0.7071068, 0.0, 0.0
    elif args.approach == "right":
        # Gripper points inward from the right (+Y direction)
        qw, qx, qy, qz = 0.0, 0.0, -0.7071068, 0.7071068
    else:
        qw, qx, qy, qz = args.qw, args.qx, args.qy, args.qz
        
    get_ik(args.x, args.y, args.z, qw, qx, qy, qz)
