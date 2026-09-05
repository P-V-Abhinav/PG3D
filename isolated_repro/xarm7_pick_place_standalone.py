"""Standalone, from-scratch xArm7-gripper pick-and-place repro.

Deliberately NOT wired into the rest of this repo: it does not import
anything from ``pg3d/``, ``scripts/``, or ``dataset_generation/``. The agent
(xArm7 + parallel-jaw gripper), the environment (table + one cube), and the
scripted pick-and-place controller are all defined in this single file, so it
can be debugged completely independently of the DP3 policy / motion planner /
reranking machinery used elsewhere. The only imports are third-party
(mani_skill, sapien, torch, numpy) plus the Python standard library.

Scene: one table (ManiSkill's stock TableSceneBuilder), one xArm7 + xArm
parallel gripper, one cube. Nothing else — no goal markers, no cameras beyond
one render camera, no obstacles.

Scripted flow (closed-loop Cartesian position servoing, not a learned
policy):
  1. reset            -> arm at its keyframe rest pose, gripper open, cube on the table.
  2. transit          -> servo XY+Z to a hover pose above the cube.
  3. descend          -> servo straight down to the cube's height.
  4. close (ramped)   -> arm target frozen; gripper target ramped open->closed
                         over N steps. This is the fix for the "arm jolts
                         during transit" failure: closing is never issued as
                         a single large instantaneous target jump, and it
                         never overlaps with arm motion.
  5. lift + transit   -> servo to a place location elsewhere on the table,
                         gripper action held at the closed target throughout.
  6. descend          -> servo down to place height.
  7. open (ramped)    -> arm target frozen; gripper target ramped closed->open.
  8. retreat          -> lift back up.

Two known physics fixes are baked into the agent config below (both found by
inspecting this project's actual custom xArm7Gripper agent and its git-log
comments, then re-derived independently here):
  * gripper_force_limit is kept low (1.0 by default, overridable via
    --gripper-force-limit) and the close/open transitions are always ramped
    with the arm frozen -- driving the mimic gripper's target from fully
    open to fully closed in a single step, especially at a high force_limit,
    slams the four-bar linkage shut on empty air and kicks the whole arm.
  * the finger links (`left_finger`/`right_finger`) are given an explicit
    high-friction contact material. The ManiSkill-bundled
    xarm7_with_gripper.urdf sets NO surface friction on these links (only
    joint friction, which is irrelevant to grip), so they silently fell back
    to SAPIEN's low default material -- meaning no amount of clamp force
    could retain a held object once the arm started moving. This is the
    actual cause of "even a perfect cube slips out during transport".

ASSUMPTIONS THIS SCRIPT COULD NOT VERIFY WITHOUT RUNNING IT (no sim
dependencies are installed on this machine -- run on the server and check the
printed diagnostics + saved video against these):
  * TableSceneBuilder's table top is assumed to sit at world z=0 (standard
    ManiSkill convention). CUBE_SPAWN_Z is set relative to that.
  * CUBE_SPAWN_XY / PLACE_XY are guesses at in-workspace points given
    ROBOT_BASE_POSE=(-0.615, 0, 0) (same base pose this repo's other xArm7
    envs use) -- adjust if the arm can't reach either point.
  * The gripper's approach orientation is NOT actively controlled -- it is
    whatever the "rest" keyframe naturally produces at agent.tcp_pose, held
    fixed for the whole episode (pd_ee_delta_pos never touches orientation,
    by construction -- see PDEEPosController.compute_target_pose). If the
    rendered video shows the gripper approaching the cube from a bad angle
    (not roughly top-down), that's the thing to fix next -- see the
    --tilt-down-deg flag for a quick interactive fix.

Run on the server, e.g.:
    python isolated_repro/xarm7_pick_place_standalone.py \\
        --gripper-force-limit 1.0 --video-out /tmp/xarm7_repro/run.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *  # noqa: F401,F403 -- controller configs + deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# ===========================================================================
# Tunable constants -- everything spatial lives here so it's easy to adjust
# after watching the first run's video, without hunting through the file.
# ===========================================================================

ROBOT_BASE_POSE = sapien.Pose(p=[-0.615, 0.0, 0.0])  # same base pose as this repo's other xArm7 envs

CUBE_HALF_SIZE = 0.02          # 4cm cube -- comfortably within the gripper's stroke
CUBE_SPAWN_XY = (-0.25, 0.0)   # world-frame XY; ASSUMED reachable, verify on first run
CUBE_SPAWN_Z = CUBE_HALF_SIZE + 0.002   # ASSUMES table top at world z=0

PLACE_XY = (-0.25, 0.2)        # world-frame XY for the place location

PRE_GRASP_HOVER_HEIGHT = 0.12  # metres above cube center for the hover waypoint
GRASP_HEIGHT_OFFSET = 0.0      # metres added to cube center Z for the descend target
LIFT_HEIGHT = 0.15             # metres above grasp height for lift/transit waypoints

GRIPPER_CLOSE_STEPS = 30       # matches the ramp length already validated for this repo's jolt fix
GRIPPER_OPEN_STEPS = 20

POS_STEP_LIMIT = 0.05          # metres/step cap for the position servo (must match pos_lower/upper below)
POS_TOLERANCE = 0.015          # metres; servo considered "arrived" below this
MAX_SERVO_STEPS = 200          # safety cap per servo call so a bad target can't hang forever


# ===========================================================================
# 1. Agent -- xArm7 + parallel-jaw gripper, defined from scratch in this file.
# ===========================================================================

# The ManiSkill-bundled xArm7+gripper URDF, resolved next to the actual
# meshes/ directory on this server (not a Python import -- just an on-disk
# asset path). Using the raw bundled URDF directly here (rather than the
# committed pg3d/envs/xarm_adapter/assets/xarm7_with_gripper_colored.urdf)
# sidesteps that file's `meshes` symlink, which is committed pointing at a
# different machine's venv path and would be broken on this server; pointing
# straight at the URDF that already lives beside the real meshes/ folder
# needs no symlink at all. The color tags that colored copy adds are purely
# cosmetic (render material only), not physics, so nothing is lost.
_XARM7_GRIPPER_MESHES_DIR = Path(
    "/home/cross-emb/abhinav.pv/success/.venv/lib/python3.10/site-packages/mani_skill/assets/robots/xarm7/meshes"
)
_XARM7_GRIPPER_URDF = str(_XARM7_GRIPPER_MESHES_DIR.parent / "xarm7_with_gripper.urdf")


@register_agent()
class IsolatedXArm7Gripper(BaseAgent):
    """xArm7 (7-DoF) + xArm parallel-jaw gripper, self-contained for this repro."""

    uid = "isolated_xarm7_gripper"
    urdf_path = _XARM7_GRIPPER_URDF

    gripper_joint_names = [
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    ]
    _gripper_mimic = {name: {"joint": "drive_joint"} for name in gripper_joint_names[1:]}

    # --- Friction fix -----------------------------------------------------
    # left_finger/right_finger are the only links that actually touch a held
    # object (the whole finger mesh is the collision geometry -- there's no
    # separate "pad" sub-link). The bundled URDF sets no surface friction on
    # them, so without this they use SAPIEN's low default material and no
    # clamp force can retain a held object once the arm accelerates.
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_finger=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
            right_finger=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
        ),
    )

    gripper_stiffness = 1e5
    gripper_damping = 2000
    gripper_force_limit = 1.0   # overridable via --gripper-force-limit; see module docstring
    gripper_friction = 1
    # Back off the drive target from the 0.85 rad hard stop so the PD spring
    # and the joint limit never fight over the same boundary point.
    _GRIPPER_LIMIT_MARGIN = 0.01
    _GRIPPER_CLOSED = 0.85 - _GRIPPER_LIMIT_MARGIN
    _GRIPPER_OPEN = 0.0

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.0, -0.4, 0.0, 0.5, 0.0, 0.9, 0.0, *([0.0] * 6)]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    arm_stiffness = 2000
    arm_damping = [100, 100, 100, 100, 100, 100, 100]
    arm_friction = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    arm_force_limit = 100

    ee_link_name = "link_tcp"

    # The gripper is a broken four-bar loop; without this the inner/outer
    # knuckles and fingers self-collide and chatter at the closed pose.
    _no_self_collision_links = [
        "link7", "link_eef", "xarm_gripper_base_link",
        "left_outer_knuckle", "left_inner_knuckle", "left_finger",
        "right_outer_knuckle", "right_inner_knuckle", "right_finger",
    ]

    def _after_init(self) -> None:
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        links_map = self.robot.links_map
        for link_name in self._no_self_collision_links:
            if link_name in links_map:
                links_map[link_name].set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def is_static(self, threshold: float = 0.2) -> torch.Tensor:
        qvel = self.robot.get_qvel()[..., :7]  # arm joints only
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def _controller_configs(self):
        # Position-only EE control. Deliberately NOT pd_ee_delta_pose: that
        # controller also needs a target orientation, and this repro doesn't
        # try to guess link_tcp's local axis convention (see module
        # docstring). PDEEPosController.compute_target_pose explicitly keeps
        # the current EE rotation unchanged, so orientation is simply
        # whatever the rest keyframe produces, frozen for the whole episode.
        pd_ee_delta_pos = PDEEPosControllerConfig(  # noqa: F405
            joint_names=self.arm_joint_names,
            pos_lower=-POS_STEP_LIMIT,
            pos_upper=POS_STEP_LIMIT,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(  # noqa: F405
            self.gripper_joint_names,
            lower=0.0,
            upper=self._GRIPPER_CLOSED,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            mimic=self._gripper_mimic,
            normalize_action=False,   # raw target radians, not normalized to [-1, 1]
        )
        return deepcopy_dict(dict(  # noqa: F405
            pd_ee_delta_pos=dict(arm=pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
        ))


# ===========================================================================
# 2. Env -- table + cube + the agent above. Nothing else.
# ===========================================================================

@register_env("Isolated-XArm7-PickCube-v0", max_episode_steps=100_000)
class IsolatedXArm7PickCubeEnv(BaseEnv):
    """Minimal repro env: table, one cube, xArm7 + parallel gripper."""

    SUPPORTED_ROBOTS = ["isolated_xarm7_gripper"]

    def __init__(self, *args: Any, robot_uids: str = "isolated_xarm7_gripper", **kwargs: Any) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, ROBOT_BASE_POSE)

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        # Built manually (rather than via mani_skill.utils.building.actors,
        # which doesn't expose a friction material) so the cube's own contact
        # friction is explicit too -- PhysX combines two surfaces' friction
        # (default: average), so a near-frictionless cube would undermine
        # even a high-friction gripper pad.
        cube_material = sapien.physx.PhysxMaterial(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[CUBE_HALF_SIZE] * 3, material=cube_material)
        builder.add_box_visual(
            half_size=[CUBE_HALF_SIZE] * 3,
            material=sapien.render.RenderMaterial(base_color=[0.1, 0.8, 0.1, 1.0]),
        )
        builder.initial_pose = sapien.Pose(p=[CUBE_SPAWN_XY[0], CUBE_SPAWN_XY[1], CUBE_SPAWN_Z])
        self.cube = builder.build(name="cube")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)

            # NOTE: at the URDF-default zeros qpos, xArm7 sits near a
            # kinematic singularity and the PD controller can blow up
            # immediately -- always reset to the "rest" keyframe instead.
            rest_qpos = self.agent.keyframes["rest"].qpos
            qpos = torch.tensor(rest_qpos, dtype=torch.float32).unsqueeze(0).expand(b, -1).clone()
            self.agent.reset(qpos)

            cube_xyz = torch.tensor(
                [CUBE_SPAWN_XY[0], CUBE_SPAWN_XY[1], CUBE_SPAWN_Z], dtype=torch.float32
            ).unsqueeze(0).expand(b, -1).clone()
            self.cube.set_pose(Pose.create_from_pq(cube_xyz))

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return []  # no policy-observation cameras needed for a scripted repro

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(eye=[0.4, -0.55, 0.55], target=[-0.25, 0.05, 0.05])
        return CameraConfig("render_camera", pose, 640, 480, 1.0, 0.01, 100)

    def evaluate(self) -> dict[str, torch.Tensor]:
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict[str, Any]) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)


# ===========================================================================
# 3. Scripted pick-and-place controller (closed-loop Cartesian servoing).
# ===========================================================================

def _to_numpy_frame(raw: Any) -> np.ndarray:
    arr = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def _log_state(env: Any, label: str) -> None:
    tcp_p = env.unwrapped.agent.tcp_pose.p[0].cpu().numpy()
    qvel = env.unwrapped.agent.robot.get_qvel()[0, :7].cpu().numpy()
    max_qvel = float(np.max(np.abs(qvel)))
    cube_p = env.unwrapped.cube.pose.p[0].cpu().numpy()
    print(
        f"  [{label}] tcp={tcp_p.round(4).tolist()}  "
        f"max|qvel|={max_qvel:.3f} rad/s  cube={cube_p.round(4).tolist()}",
        flush=True,
    )


def _servo_to_position(
    env: Any,
    target_pos_world: np.ndarray,
    *,
    gripper_val: float,
    max_steps: int,
    pos_tol: float,
    frames: list[np.ndarray],
    max_qvel_tracker: list[float],
) -> tuple[bool, float]:
    """Closed-loop position-only servo of the TCP toward a world-frame target.

    Orientation is left untouched by construction (pd_ee_delta_pos never
    changes it). Recomputes the error from the actual sim state every step,
    so this is self-correcting rather than open-loop.
    """
    dist = float("inf")
    for _ in range(max_steps):
        tcp_p = env.unwrapped.agent.tcp_pose.p[0].cpu().numpy()
        err = np.asarray(target_pos_world, dtype=np.float32) - tcp_p
        dist = float(np.linalg.norm(err))
        if dist < pos_tol:
            return True, dist
        raw_delta = np.clip(err, -POS_STEP_LIMIT, POS_STEP_LIMIT)
        norm_delta = raw_delta / POS_STEP_LIMIT  # normalize to [-1, 1] for normalize_action=True
        action = np.concatenate([norm_delta, [gripper_val]]).astype(np.float32)
        env.step(action)
        frames.append(_to_numpy_frame(env.render()))
        qvel = env.unwrapped.agent.robot.get_qvel()[0, :7].cpu().numpy()
        max_qvel_tracker[0] = max(max_qvel_tracker[0], float(np.max(np.abs(qvel))))
    return False, dist


def _ramp_gripper(
    env: Any,
    *,
    from_val: float,
    to_val: float,
    steps: int,
    frames: list[np.ndarray],
    max_qvel_tracker: list[float],
) -> None:
    """Ramp the gripper's mimic drive target over `steps` control steps while
    holding the arm's Cartesian target fixed (zero position delta each step).

    This is the jolt fix: the drive target never jumps from open to closed
    (or back) in a single step, and it's never issued while the arm is also
    receiving a nonzero motion command.
    """
    for i in range(1, steps + 1):
        val = from_val + (to_val - from_val) * (i / steps)
        action = np.array([0.0, 0.0, 0.0, val], dtype=np.float32)
        env.step(action)
        frames.append(_to_numpy_frame(env.render()))
        qvel = env.unwrapped.agent.robot.get_qvel()[0, :7].cpu().numpy()
        max_qvel_tracker[0] = max(max_qvel_tracker[0], float(np.max(np.abs(qvel))))


def _save_video(frames: list[np.ndarray], path: str) -> None:
    if not frames:
        print("[warn] no frames recorded, nothing to save.")
        return
    try:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore[no-redef]
    except ImportError as exc:
        print(
            f"[warn] imageio not installed ({exc}); skipping video save. "
            f"{len(frames)} frames were rendered but not written to disk."
        )
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=30)
    print(f"saved video: {path}  ({len(frames)} frames)")


def main(argv: list[str] | None = None) -> int:
    import gymnasium as gym

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gripper-force-limit", type=float, default=1.0)
    parser.add_argument("--gripper-close-steps", type=int, default=GRIPPER_CLOSE_STEPS)
    parser.add_argument("--gripper-open-steps", type=int, default=GRIPPER_OPEN_STEPS)
    parser.add_argument("--max-servo-steps", type=int, default=MAX_SERVO_STEPS)
    parser.add_argument("--pos-tol", type=float, default=POS_TOLERANCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-out", type=str, default="./isolated_repro_output.mp4")
    args = parser.parse_args(argv)

    IsolatedXArm7Gripper.gripper_force_limit = args.gripper_force_limit
    print(f"gripper_force_limit = {IsolatedXArm7Gripper.gripper_force_limit}")

    env = gym.make(
        "Isolated-XArm7-PickCube-v0",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    obs, info = env.reset(seed=args.seed)

    frames: list[np.ndarray] = [_to_numpy_frame(env.render())]
    max_qvel_tracker = [0.0]  # boxed float so helper functions can update it in place

    cube_pos0 = env.unwrapped.cube.pose.p[0].cpu().numpy()
    print(f"cube spawned at {cube_pos0.tolist()}")
    _log_state(env, "reset")

    pregrasp = cube_pos0 + np.array([0.0, 0.0, PRE_GRASP_HOVER_HEIGHT], dtype=np.float32)
    grasp = cube_pos0 + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET], dtype=np.float32)
    lift = grasp + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32)
    place_hover = np.array([PLACE_XY[0], PLACE_XY[1], lift[2]], dtype=np.float32)
    place_down = np.array([PLACE_XY[0], PLACE_XY[1], grasp[2]], dtype=np.float32)

    print("\n--- [phase 1] transit: hover above cube ---")
    ok, dist = _servo_to_position(
        env, pregrasp, gripper_val=IsolatedXArm7Gripper._GRIPPER_OPEN,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m  max|qvel| so far={max_qvel_tracker[0]:.3f} rad/s")
    _log_state(env, "post-hover")

    print("\n--- [phase 2] descend to cube ---")
    ok, dist = _servo_to_position(
        env, grasp, gripper_val=IsolatedXArm7Gripper._GRIPPER_OPEN,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m  max|qvel| so far={max_qvel_tracker[0]:.3f} rad/s")
    _log_state(env, "post-descend")
    transit_max_qvel = max_qvel_tracker[0]

    print(f"\n--- [phase 3] ramped close over {args.gripper_close_steps} steps (arm frozen) ---")
    _ramp_gripper(
        env, from_val=IsolatedXArm7Gripper._GRIPPER_OPEN, to_val=IsolatedXArm7Gripper._GRIPPER_CLOSED,
        steps=args.gripper_close_steps, frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    close_jolt_qvel = max_qvel_tracker[0]
    print(f"  max|qvel| during close = {close_jolt_qvel:.3f} rad/s (compare to transit's {transit_max_qvel:.3f})")
    _log_state(env, "post-close")

    print("\n--- [phase 4] lift ---")
    ok, dist = _servo_to_position(
        env, lift, gripper_val=IsolatedXArm7Gripper._GRIPPER_CLOSED,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m")
    _log_state(env, "post-lift")
    lift_cube_z = env.unwrapped.cube.pose.p[0, 2].item()

    print("\n--- [phase 5] transit to place location (gripper held closed) ---")
    ok, dist = _servo_to_position(
        env, place_hover, gripper_val=IsolatedXArm7Gripper._GRIPPER_CLOSED,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m  max|qvel| so far={max_qvel_tracker[0]:.3f} rad/s")
    _log_state(env, "post-transport")
    transport_cube_z = env.unwrapped.cube.pose.p[0, 2].item()
    print(
        f"  cube height right after lift: {lift_cube_z:.4f} m -> after transport: {transport_cube_z:.4f} m "
        f"(a large drop here means the cube slipped out mid-transit)"
    )

    print("\n--- [phase 6] descend to place height ---")
    ok, dist = _servo_to_position(
        env, place_down, gripper_val=IsolatedXArm7Gripper._GRIPPER_CLOSED,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m")
    _log_state(env, "post-place-descend")

    print(f"\n--- [phase 7] ramped release over {args.gripper_open_steps} steps (arm frozen) ---")
    _ramp_gripper(
        env, from_val=IsolatedXArm7Gripper._GRIPPER_CLOSED, to_val=IsolatedXArm7Gripper._GRIPPER_OPEN,
        steps=args.gripper_open_steps, frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    _log_state(env, "post-release")

    print("\n--- [phase 8] retreat ---")
    _servo_to_position(
        env, place_down + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32),
        gripper_val=IsolatedXArm7Gripper._GRIPPER_OPEN,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    _log_state(env, "final")

    final_cube_pos = env.unwrapped.cube.pose.p[0].cpu().numpy()
    xy_offset_from_place = float(np.linalg.norm(final_cube_pos[:2] - np.asarray(PLACE_XY)))
    print(f"\nfinal cube position: {final_cube_pos.tolist()}")
    print(f"place target XY was: {list(PLACE_XY)}  (final cube XY offset from it: {xy_offset_from_place:.4f} m)")
    print(f"overall max|qvel| observed (any phase): {max_qvel_tracker[0]:.3f} rad/s")
    print("  > 5-10 rad/s sustained is already suspicious; ~100 rad/s is the jolt failure mode seen before.")

    _save_video(frames, args.video_out)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
