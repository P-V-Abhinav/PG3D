#!/usr/bin/env python
"""Wrapper to write a pg3d reach multimodal pose dataset for the xArm7 with parallel-jaw gripper.

Automatically injects the required XARM7 workspace bounds, gripper-open settings,
and environment ID so you don't have to specify them manually.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path so dataset_generation can be imported if running directly
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from dataset_generation.write_xarm7_gripper_reach_dataset import _XARM7_GRIPPER_DEFAULTS, _bounds_flag
from pg3d.envs.xarm_adapter.reach_config import XARM7_REACH_WORKSPACE_BOUNDS, XARM7_WORKSPACE_BOUNDS
from scripts.write_maniskill_pose_reach_dataset import main as pose_main

def main() -> int:
    # Extract user arguments, excluding the script name
    user_argv = list(sys.argv[1:])
    
    injected = (
        _XARM7_GRIPPER_DEFAULTS
        + _bounds_flag("--reach-workspace-bounds", XARM7_REACH_WORKSPACE_BOUNDS)
        + _bounds_flag("--workspace-bounds", XARM7_WORKSPACE_BOUNDS)
    )
    
    print(f"Injecting XArm7 defaults: {injected}")
    
    # Run the multimodal pose generator with the injected defaults + any user overrides
    return pose_main(injected + user_argv)

if __name__ == "__main__":
    raise SystemExit(main())
