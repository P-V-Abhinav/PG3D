"""Standalone, from-scratch xArm7-gripper pick-and-place repro (v2 -- grasp
pose rewritten from ManiSkill's own documented grasp-pose convention).

Deliberately NOT wired into the rest of this repo: it does not import
anything from ``pg3d/``, ``scripts/``, or ``dataset_generation/``. The agent
(xArm7 + parallel-jaw gripper), the environment (table + one cube), and the
scripted pick-and-place controller are all defined in this single file. The
only imports are third-party (mani_skill, sapien, torch, numpy) plus the
Python standard library.

WHY THIS VERSION IS A FROM-SCRATCH REWRITE OF THE GRASP LOGIC
================================================================
The previous version servoed the TCP to a position directly above the cube
but never controlled ORIENTATION -- it just left the gripper at whatever
rotation the "rest" keyframe happened to produce, flagged as an unverified
assumption. Running it showed the actual consequence: the gripper closed
"successfully" (ramp completed with no stall) but the cube was kicked ~6cm
sideways during the close and the arm itself got shoved ~4cm off target --
i.e. the fingers were not actually straddling the cube symmetrically, so
closing swiped/flicked it instead of trapping it. Bad orientation, not gripper
force, was the real cause of that failure.

This version fixes that at the root by computing a real, geometrically
correct grasp orientation instead of leaving it uncontrolled, using the exact
same convention ManiSkill's own official scripted solutions use (see
mani_skill.examples.motionplanning.xarm6.solutions.pick_cube -- read locally
on this machine for reference, not imported by this script):

  1. Pick a world-frame `approaching` axis (the direction the gripper's TCP
     frame should point *into* the object -- [0, 0, -1] for a straight
     top-down grasp) and a `closing` axis (the world direction along which
     the two fingers should straddle the object -- for a cube, any axis
     orthogonal to `approaching` works, since a cube's cross-section is
     identical from every side; ManiSkill's own reference solution instead
     derives this from the object's oriented bounding box, which matters for
     non-cube objects but is unnecessary complexity here).
  2. Build a target rotation matrix R = [ortho, closing, approaching] as
     columns, where ortho = closing x approaching. This is IDENTICAL to the
     `build_grasp_pose` staticmethod defined on every ManiSkill two-finger
     gripper agent (Panda, xarm6_robotiq, Fetch, SO-100 all use this exact
     formula) -- reimplemented here as plain numpy (see
     `_grasp_rotation_matrix` below) rather than imported, since this file
     imports no code from any existing agent class.
  3. Verify this convention actually matches *this* gripper's `link_tcp`
     frame by reading the raw URDF geometry (not assuming it): `link_tcp` is
     a fixed child of `xarm_gripper_base_link` with zero rotation offset
     (origin rpy="0 0 0"), so it shares that link's axes exactly. The
     drive_joint/right_outer_knuckle_joint origins place the left finger
     assembly at +Y and the right at -Y of that frame, both rotating about
     local X -- i.e. the fingers separate along local Y ("closing" = local Y)
     and extend outward along local Z ("approaching" = local Z), which is
     exactly the column order in the R above. So this convention is
     confirmed correct for this specific gripper, not assumed by analogy.

Given a correct (position, orientation) grasp pose, the servo now drives the
TCP there using `pd_ee_pose_abs` -- a PDEEPoseController configured with
`use_delta=False, normalize_action=False`. This was deliberately chosen over
the delta-controller used previously: with `normalize_action=False` the
action IS the literal absolute target ([x, y, z, roll, pitch, yaw] in the
robot's root frame, root frame = world frame here since ROBOT_BASE_POSE has
no rotation) with no [-1, 1] rescaling, and no delta/frame-composition
semantics to get subtly wrong. The one piece that still had to be gotten
exactly right -- which Euler convention `action[3:6]` is decoded with -- is
resolved exactly (not approximated) by using mani_skill's own
`matrix_to_euler_angles(..., "XYZ")`, the documented, exact inverse of the
`euler_angles_to_matrix(..., "XYZ")` this controller decodes the action with
(both in mani_skill.utils.geometry.rotation_conversions -- a mani_skill
library module, not a copy of any file in this repo).

Scene: one table (ManiSkill's stock TableSceneBuilder), one xArm7 + xArm
parallel gripper, one cube. Nothing else.

Scripted flow (closed-loop 6-DoF Cartesian pose servoing, not a learned
policy) -- the grasp orientation computed once above is held fixed for the
whole episode, only the target position changes between phases:
  1. reset            -> arm at its keyframe rest pose, gripper open, cube on the table.
  2. transit          -> servo to a hover pose above the cube (grasp orientation).
  3. descend          -> servo straight down to the cube's height (same orientation).
  4. close-until-stall-> arm target frozen at the grasp pose; gripper target
                         ramped open->closed, but the ramp stops the moment the
                         drive joint stalls against the object (near-zero qvel
                         while still short of the ramp's current target) --
                         freezing there means the PD controller only ever
                         supplies a *holding* force afterwards, not a
                         continuously growing command to close through a
                         rigid object it physically can't reach.
  5. lift + transit   -> servo to a place location elsewhere on the table,
                         gripper action held at the frozen contact target.
  6. descend          -> servo down to place height.
  7. open (ramped)    -> arm target frozen; gripper ramped closed->open.
  8. retreat          -> lift back up.

ASSUMPTIONS THIS SCRIPT COULD NOT VERIFY WITHOUT RUNNING IT:
  * TableSceneBuilder's table top is assumed to sit at world z=0 (standard
    ManiSkill convention). CUBE_SPAWN_Z is set relative to that.
  * CUBE_SPAWN_XY / PLACE_XY are guesses at in-workspace points given
    ROBOT_BASE_POSE=(-0.615, 0, 0) (same base pose this repo's other xArm7
    envs use) -- adjust if the arm can't reach either point.
  * The gripper's own contact-friction/force config (kept from the previous
    version -- these were separately validated against the "cube slips
    during transport" and "arm jolts during close" failures, which are
    independent of the orientation bug this rewrite targets).

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
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# ===========================================================================
# Tunable constants -- everything spatial lives here so it's easy to adjust
# after watching a run's video, without hunting through the file.
# ===========================================================================

ROBOT_BASE_POSE = sapien.Pose(p=[-0.615, 0.0, 0.0])  # same base pose as this repo's other xArm7 envs

CUBE_HALF_SIZE = 0.02          # 4cm cube -- comfortably within the gripper's stroke
CUBE_SPAWN_XY = (-0.25, 0.0)   # world-frame XY; ASSUMED reachable, verify on first run
CUBE_SPAWN_Z = CUBE_HALF_SIZE + 0.002   # ASSUMES table top at world z=0

PLACE_XY = (-0.25, 0.4)        # world-frame XY for the place location -- doubled from the
                                # previous 0.2 (which the last run reached with plenty of
                                # residual reach margin, ~0.5m from ROBOT_BASE_POSE either
                                # way) to stress-test grip stability over a longer transport

# Grasp axes (world frame) -- see module docstring for why these are correct
# for THIS gripper's link_tcp frame, verified from URDF joint geometry.
GRASP_APPROACHING = np.array([0.0, 0.0, -1.0])   # straight top-down
GRASP_CLOSING = np.array([1.0, 0.0, 0.0])        # arbitrary axis orthogonal to
                                                  # approaching -- fine for a
                                                  # cube, whose cross-section
                                                  # is identical from every side

PRE_GRASP_HOVER_HEIGHT = 0.12  # metres above cube center for the hover waypoint
GRASP_HEIGHT_OFFSET = 0.0      # metres added to cube center Z for the descend target
LIFT_HEIGHT = 0.15             # metres above grasp height for lift/transit waypoints

GRIPPER_CLOSE_STEPS = 30       # max ramp length for the stall-detected close
GRIPPER_OPEN_STEPS = 20

POS_TOLERANCE = 0.015          # metres; servo considered "arrived" below this
MAX_SERVO_STEPS = 200          # safety cap per servo call so a bad target can't hang forever


def _grasp_rotation_matrix(approaching: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Build a target TCP rotation matrix from world-frame approach/closing axes.

    Identical formula to the `build_grasp_pose` staticmethod ManiSkill defines
    on every one of its two-finger-gripper agent classes (Panda, xarm6_robotiq,
    Fetch, SO-100) -- reimplemented here as plain numpy rather than imported,
    since this file imports no code from any existing agent class. Columns of
    the returned matrix are [ortho, closing, approaching]: local X = ortho
    (closing x approaching), local Y = closing (finger separation axis),
    local Z = approaching (the axis pointing from the gripper into the
    object). See the module docstring for why this column order matches this
    specific gripper's `link_tcp` frame (verified from its URDF geometry, not
    assumed by analogy to Panda/xarm6_robotiq).
    """
    approaching = approaching / np.linalg.norm(approaching)
    closing = closing - (approaching @ closing) * approaching  # orthogonalize
    closing = closing / np.linalg.norm(closing)
    ortho = np.cross(closing, approaching)
    return np.stack([ortho, closing, approaching], axis=1)  # (3, 3), columns as above


def _rotation_matrix_to_root_euler_xyz(rot_matrix: np.ndarray) -> np.ndarray:
    """World-frame rotation matrix -> the exact XYZ-Euler triple `pd_ee_pose_abs`
    expects in action[3:6], using mani_skill's own exact inverse of the
    matrix<->euler conversion that controller decodes actions with (so this
    is exact, not an approximation of the controller's convention).
    """
    mat = torch.as_tensor(rot_matrix, dtype=torch.float32).unsqueeze(0)
    euler = matrix_to_euler_angles(mat, "XYZ")[0]
    return euler.numpy()


# ===========================================================================
# 1. Agent -- xArm7 + parallel-jaw gripper, defined from scratch in this file.
# ===========================================================================

# xArm7+gripper URDF resolution.
#
# The raw ManiSkill-bundled xarm7_with_gripper.urdf is NOT actually present
# on this server (only its meshes/ directory is -- the URDF itself is gated
# behind ManiSkill's interactive asset-download prompt, confirmed by running
# this script). And the git-committed
# pg3d/envs/xarm_adapter/assets/xarm7_with_gripper_colored.urdf *is* present,
# but its own `meshes` symlink is committed pointing at a different machine's
# venv path, so it's broken here too.
#
# Fix: read that committed URDF's TEXT (a plain file read, not a Python
# import -- no code from agents.py is executed) and rewrite every relative
# "meshes/..." reference to an absolute path under the real meshes/
# directory on this server, then write the patched copy next to this
# script. This needs no symlink and no asset download. The color tags the
# committed copy adds are purely cosmetic (render material only, not
# physics), so reusing it changes nothing physically.
_XARM7_GRIPPER_MESHES_DIR = Path(
    "/home/cross-emb/abhinav.pv/success/.venv/lib/python3.10/site-packages/mani_skill/assets/robots/xarm7/meshes"
)


def _resolve_gripper_urdf() -> str:
    src = (
        Path(__file__).resolve().parent.parent
        / "pg3d" / "envs" / "xarm_adapter" / "assets" / "xarm7_with_gripper_colored.urdf"
    )
    if not src.exists():
        raise FileNotFoundError(
            f"expected the committed URDF at {src}. If this checkout doesn't have it, "
            "point _resolve_gripper_urdf() at some other xarm7-with-gripper URDF you do have."
        )
    if not _XARM7_GRIPPER_MESHES_DIR.is_dir():
        raise FileNotFoundError(
            f"_XARM7_GRIPPER_MESHES_DIR does not exist: {_XARM7_GRIPPER_MESHES_DIR}"
        )
    text = src.read_text()
    patched = text.replace('filename="meshes/', f'filename="{_XARM7_GRIPPER_MESHES_DIR}/')
    if patched == text:
        raise RuntimeError(f"no 'meshes/...' mesh references found in {src} -- unexpected URDF format.")
    dst = Path(__file__).resolve().parent / "_generated_xarm7_with_gripper.urdf"
    dst.write_text(patched)
    return str(dst)


_XARM7_GRIPPER_URDF = _resolve_gripper_urdf()


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

    # --- Friction fix -------------------------------------------------------
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

    # --- Vibration fix -------------------------------------------------
    # force = stiffness*(target - qpos) - damping*qvel, clamped to
    # force_limit. At stiffness=1e5 (this repo's original value, matched to
    # ManiSkill's own xarm6_robotiq mimic gripper), the force saturates the
    # instant the position error exceeds force_limit/stiffness = 1/1e5 =
    # 0.00001 rad -- i.e. essentially any nonzero error at all. Once the
    # fingers hold solid contact, that turns the "spring" into a relay: the
    # smallest contact deflection commands max force the other way,
    # overshoots, flips sign, and repeats -- a limit-cycle chatter, which is
    # exactly the "gripper teeth vibrating" symptom. Lowering stiffness (and
    # damping proportionally) gives the spring a real proportional band
    # before it saturates, so it settles into a compliant hold instead of
    # oscillating. force_limit stays modest (max sustained holding force is
    # unchanged in magnitude) -- only how "hard" the spring is near the
    # setpoint changes.
    gripper_stiffness = 400
    gripper_damping = 40
    gripper_force_limit = 1.0   # overridable via --gripper-force-limit
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
        # Absolute 6-DoF EE pose control: use_delta=False + normalize_action=False
        # means the action IS the literal target [x, y, z, roll, pitch, yaw] in
        # the robot's root frame (== world frame here, since ROBOT_BASE_POSE has
        # no rotation) -- no [-1, 1] rescaling, no delta/frame-composition
        # semantics. See module docstring for why this was chosen over a delta
        # controller.
        pd_ee_pose_abs = PDEEPoseControllerConfig(  # noqa: F405
            joint_names=self.arm_joint_names,
            pos_lower=-2.0,
            pos_upper=2.0,
            rot_lower=-2 * np.pi,
            rot_upper=2 * np.pi,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
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
            pd_ee_pose_abs=dict(arm=pd_ee_pose_abs, gripper=gripper_pd_joint_pos),
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

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        # Scripted repro, no learned reward needed -- but BaseEnv.get_reward
        # calls this by default (reward_mode="normalized_dense"), and the
        # base class's version just raises NotImplementedError.
        return torch.zeros(self.num_envs, device=self.device)


# ===========================================================================
# 3. Scripted pick-and-place controller (closed-loop 6-DoF pose servoing).
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


def _pose_action(target_pos_world: np.ndarray, target_euler_xyz: np.ndarray, gripper_val: float) -> np.ndarray:
    """Assemble the 7-dim pd_ee_pose_abs+gripper action: absolute [x,y,z,
    roll,pitch,yaw] in root frame (== world frame here, ROBOT_BASE_POSE has
    no rotation) followed by the gripper's raw target radians.
    """
    target_pos_root = np.asarray(target_pos_world, dtype=np.float32) - np.asarray(ROBOT_BASE_POSE.p, dtype=np.float32)
    return np.concatenate(
        [target_pos_root, np.asarray(target_euler_xyz, dtype=np.float32), [np.float32(gripper_val)]]
    ).astype(np.float32)


def _servo_to_pose(
    env: Any,
    target_pos_world: np.ndarray,
    target_euler_xyz: np.ndarray,
    *,
    gripper_val: float,
    max_steps: int,
    pos_tol: float,
    frames: list[np.ndarray],
    max_qvel_tracker: list[float],
) -> tuple[bool, float]:
    """Closed-loop servo of the TCP toward a fixed absolute (position,
    orientation) target. The action itself is constant every step (this is
    an absolute, non-delta controller -- see module docstring); repeated
    stepping is what lets the physically simulated PD/IK dynamics actually
    converge the arm to that target over multiple substeps.
    """
    action = _pose_action(target_pos_world, target_euler_xyz, gripper_val)
    dist = float("inf")
    for _ in range(max_steps):
        tcp_p = env.unwrapped.agent.tcp_pose.p[0].cpu().numpy()
        dist = float(np.linalg.norm(np.asarray(target_pos_world, dtype=np.float32) - tcp_p))
        if dist < pos_tol:
            return True, dist
        env.step(action)
        frames.append(_to_numpy_frame(env.render()))
        qvel = env.unwrapped.agent.robot.get_qvel()[0, :7].cpu().numpy()
        max_qvel_tracker[0] = max(max_qvel_tracker[0], float(np.max(np.abs(qvel))))
    return False, dist


def _ramp_gripper(
    env: Any,
    *,
    frozen_pos_world: np.ndarray,
    frozen_euler_xyz: np.ndarray,
    from_val: float,
    to_val: float,
    steps: int,
    frames: list[np.ndarray],
    max_qvel_tracker: list[float],
) -> None:
    """Ramp the gripper's mimic drive target over `steps` control steps while
    holding the arm's pose target fixed at `frozen_pos_world`/`frozen_euler_xyz`.

    This is the jolt fix: the drive target never jumps from open to closed
    (or back) in a single step, and the arm's own target never changes while
    the gripper is being ramped.
    """
    for i in range(1, steps + 1):
        val = from_val + (to_val - from_val) * (i / steps)
        action = _pose_action(frozen_pos_world, frozen_euler_xyz, val)
        env.step(action)
        frames.append(_to_numpy_frame(env.render()))
        qvel = env.unwrapped.agent.robot.get_qvel()[0, :7].cpu().numpy()
        max_qvel_tracker[0] = max(max_qvel_tracker[0], float(np.max(np.abs(qvel))))


def _close_gripper_until_contact(
    env: Any,
    *,
    frozen_pos_world: np.ndarray,
    frozen_euler_xyz: np.ndarray,
    target_val: float,
    max_ramp_steps: int,
    frames: list[np.ndarray],
    max_qvel_tracker: list[float],
    qvel_stall_thresh: float = 0.05,
    stall_patience: int = 5,
    settle_steps: int = 20,  # bumped from 5 so a softer spring (post vibration-fix) has time to visibly settle
) -> float:
    """Ramp the gripper's drive_joint target toward `target_val`, but STOP
    advancing it as soon as the joint stalls against something (qvel near
    zero for `stall_patience` consecutive steps while a real position gap to
    the commanded target still remains), then hold the target frozen there.

    Freezing the target at first contact means the PD controller only ever
    supplies enough restoring force to hold that position afterwards, not a
    continuously-growing command to close through a rigid object.

    Uses the drive_joint's own qpos/qvel (looked up by name via
    active_joints_map, not a positional index into the flat qpos vector --
    robust regardless of how mani_skill orders DOFs for this articulation).

    Returns the held target value (float) -- pass this as `gripper_val` for
    every subsequent action instead of `target_val`.
    """
    drive_joint = env.unwrapped.agent.robot.active_joints_map["drive_joint"]
    step_size = target_val / max_ramp_steps
    current_target = 0.0
    stall_count = 0
    contact_step = None

    for i in range(1, max_ramp_steps + 1):
        drive_qpos = float(drive_joint.qpos[0])
        drive_qvel = float(drive_joint.qvel[0])

        if abs(drive_qvel) < qvel_stall_thresh and (current_target - drive_qpos) > 0.02:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= stall_patience:
            contact_step = i
            break

        current_target = min(target_val, current_target + step_size)
        action = _pose_action(frozen_pos_world, frozen_euler_xyz, current_target)
        env.step(action)
        frames.append(_to_numpy_frame(env.render()))
        qvel = env.unwrapped.agent.robot.get_qvel()[0, :7].cpu().numpy()
        max_qvel_tracker[0] = max(max_qvel_tracker[0], float(np.max(np.abs(qvel))))

    if contact_step is not None:
        print(
            f"    [gripper] contact stall detected at ramp step {contact_step}: "
            f"drive_qpos={float(drive_joint.qpos[0]):.4f}  "
            f"held target={current_target:.4f}  (commanded {target_val:.4f})"
        )
    else:
        print(
            f"    [gripper] ramp completed with no contact stall detected -- "
            f"either the cube wasn't between the fingers, or stall_thresh/patience "
            f"need tuning. held target={current_target:.4f} (commanded {target_val:.4f})"
        )

    for _ in range(settle_steps):
        action = _pose_action(frozen_pos_world, frozen_euler_xyz, current_target)
        env.step(action)
        frames.append(_to_numpy_frame(env.render()))
        qvel = env.unwrapped.agent.robot.get_qvel()[0, :7].cpu().numpy()
        max_qvel_tracker[0] = max(max_qvel_tracker[0], float(np.max(np.abs(qvel))))

    return current_target


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
    parser.add_argument("--gripper-stall-qvel-thresh", type=float, default=0.05)
    parser.add_argument("--gripper-stall-patience", type=int, default=5)
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
        control_mode="pd_ee_pose_abs",
        render_mode="rgb_array",
        num_envs=1,
    )
    obs, info = env.reset(seed=args.seed)

    frames: list[np.ndarray] = [_to_numpy_frame(env.render())]
    max_qvel_tracker = [0.0]  # boxed float so helper functions can update it in place

    cube_pos0 = env.unwrapped.cube.pose.p[0].cpu().numpy()
    print(f"cube spawned at {cube_pos0.tolist()}")
    _log_state(env, "reset")

    # --- Grasp pose: computed once, held fixed for the whole episode -------
    grasp_rot = _grasp_rotation_matrix(GRASP_APPROACHING, GRASP_CLOSING)
    grasp_euler = _rotation_matrix_to_root_euler_xyz(grasp_rot)
    print(
        f"grasp orientation: approaching={GRASP_APPROACHING.tolist()} "
        f"closing={GRASP_CLOSING.tolist()} -> euler_xyz={grasp_euler.round(4).tolist()}"
    )

    pregrasp = cube_pos0 + np.array([0.0, 0.0, PRE_GRASP_HOVER_HEIGHT], dtype=np.float32)
    grasp = cube_pos0 + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET], dtype=np.float32)
    lift = grasp + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32)
    place_hover = np.array([PLACE_XY[0], PLACE_XY[1], lift[2]], dtype=np.float32)
    place_down = np.array([PLACE_XY[0], PLACE_XY[1], grasp[2]], dtype=np.float32)

    print("\n--- [phase 1] transit: hover above cube ---")
    ok, dist = _servo_to_pose(
        env, pregrasp, grasp_euler, gripper_val=IsolatedXArm7Gripper._GRIPPER_OPEN,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m  max|qvel| so far={max_qvel_tracker[0]:.3f} rad/s")
    _log_state(env, "post-hover")

    print("\n--- [phase 2] descend to cube ---")
    ok, dist = _servo_to_pose(
        env, grasp, grasp_euler, gripper_val=IsolatedXArm7Gripper._GRIPPER_OPEN,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m  max|qvel| so far={max_qvel_tracker[0]:.3f} rad/s")
    _log_state(env, "post-descend")
    transit_max_qvel = max_qvel_tracker[0]

    print(f"\n--- [phase 3] close until contact (stall-detected, arm frozen) ---")
    held_closed_val = _close_gripper_until_contact(
        env, frozen_pos_world=grasp, frozen_euler_xyz=grasp_euler,
        target_val=IsolatedXArm7Gripper._GRIPPER_CLOSED,
        max_ramp_steps=args.gripper_close_steps,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
        qvel_stall_thresh=args.gripper_stall_qvel_thresh,
        stall_patience=args.gripper_stall_patience,
    )
    close_jolt_qvel = max_qvel_tracker[0]
    print(f"  max|qvel| during close = {close_jolt_qvel:.3f} rad/s (compare to transit's {transit_max_qvel:.3f})")
    _log_state(env, "post-close")

    print("\n--- [phase 4] lift ---")
    ok, dist = _servo_to_pose(
        env, lift, grasp_euler, gripper_val=held_closed_val,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m")
    _log_state(env, "post-lift")
    lift_cube_z = env.unwrapped.cube.pose.p[0, 2].item()

    print("\n--- [phase 5] transit to place location (gripper held at contact target) ---")
    ok, dist = _servo_to_pose(
        env, place_hover, grasp_euler, gripper_val=held_closed_val,
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
    ok, dist = _servo_to_pose(
        env, place_down, grasp_euler, gripper_val=held_closed_val,
        max_steps=args.max_servo_steps, pos_tol=args.pos_tol,
        frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    print(f"  arrived={ok}  residual={dist:.4f} m")
    _log_state(env, "post-place-descend")

    print(f"\n--- [phase 7] ramped release over {args.gripper_open_steps} steps (arm frozen) ---")
    _ramp_gripper(
        env, frozen_pos_world=place_down, frozen_euler_xyz=grasp_euler,
        from_val=held_closed_val, to_val=IsolatedXArm7Gripper._GRIPPER_OPEN,
        steps=args.gripper_open_steps, frames=frames, max_qvel_tracker=max_qvel_tracker,
    )
    _log_state(env, "post-release")

    print("\n--- [phase 8] retreat ---")
    _servo_to_pose(
        env, place_down + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32), grasp_euler,
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
