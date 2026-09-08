"""master_env_settings.py -- the one file that differs between repo copies.

The eval suite is mirrored into more than one checkout. Every other module in
this package is byte-identical across those copies; this module is the seam that
binds the suite to whichever repo it sits in, so a difference between checkouts
can only ever live here.

**This is the OLD-repo (local testing) version.** This repo's own
``reach_env``/``reach_config`` still carry the pre-M2 layout, so binding
straight to them would silently evaluate a different physical setup than the
official one. This seam therefore re-declares master's settings explicitly and
pins them on top of the local env:

1. **Base at the world origin** (this repo bolts the arm at ``[-0.615, 0, 0]``),
   with the compensating ``+0.615 m`` table shift re-applied every episode --
   ADR 0014 / ADR 0022. Same robot-to-table arrangement as master.
2. **Master's eye-on-base camera calibration** (this repo still has the earlier
   eye-to-hand calibration at ``[1.7103, 0.0043, 0.7097]``), at master's
   aspect-matched ``164 x 96`` render and 58 deg vertical FOV. The per-episode
   jitter and the calibration-error model are re-implemented here against
   master's nominal pose, with the same magnitudes (+/-10 cm, +/-2 deg,
   9.5284 mm / 1.4328 deg RMS).
3. **Exactly one sensor camera**, as master declares: ``base_camera``. This
   repo's gripper reach env mounts a second one (``cam_wrist`` on ``link_tcp``)
   that master does not have, and ManiSkill fuses every sensor camera into one
   point cloud, so a second camera would change the observation the policy is
   fed. It is dropped here.
4. **master's human render camera.** This repo aims its third-person camera at
   the M1 workspace, which now points past the robot at empty space.
5. **This repo's gripper conventions.** Its ``XArm7Gripper`` already exposes the
   commandable mimic gripper under ``pd_joint_pos`` (master needs
   ``pd_joint_pos_gripper``; plain ``pd_joint_pos`` there is a static closed
   hold), so the control mode a pick task asks for is named differently.

The master repo is the single source of truth for all of these numbers. If a
calibration changes there, this file has to be updated by hand -- which is
exactly why the official runs are done from the master checkout and this copy
exists only for local testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation


# Every SAPIEN / ManiSkill import in this module stays inside a function body,
# the same rule `pg3d.tasks.live` follows: it keeps `eval_config` -- which needs
# only the numeric constants below -- importable (and its frozen numbers
# auditable) on a machine with no simulator installed.

# ---------------------------------------------------------------------------
# Frame and bounds (master values, world frame with the base at the origin)
# ---------------------------------------------------------------------------
FRAME_SHIFT: np.ndarray = np.array([0.615, 0.0, 0.0], dtype=np.float32)

#: master `XARM7_CROP_BOX_BASE` + base at the origin.
CROP_BOUNDS: np.ndarray = np.array(
    [[-0.15, 0.79], [-0.55, 0.55], [-0.02, 0.70]], dtype=np.float32
)

#: master `XARM7_REACH_BOX_BASE` + base at the origin.
REACH_BOUNDS: np.ndarray = np.array(
    [[0.18, 0.50], [-0.42, 0.42], [0.05, 0.37]], dtype=np.float32
)

#: Where master's human render camera looks: the centre of the reach box. The
#: eye sits 1.0 m back and 0.25 m above it. Viz only -- no policy input -- but a
#: mis-aimed one makes every eval video useless.
RENDER_CAMERA_TARGET: tuple[float, float, float] = (
    float((REACH_BOUNDS[0, 0] + REACH_BOUNDS[0, 1]) / 2),
    float((REACH_BOUNDS[1, 0] + REACH_BOUNDS[1, 1]) / 2),
    float((REACH_BOUNDS[2, 0] + REACH_BOUNDS[2, 1]) / 2),
)

#: This repo's gripper mimic controller lives under `pd_joint_pos`.
GRIPPER_CONTROL_MODE = "pd_joint_pos"

# ---------------------------------------------------------------------------
# master camera (eye-on-base calibration, `xarm_rs_on_base_calibration`)
# ---------------------------------------------------------------------------
ROBOT_BASE_POSITION = np.zeros(3, dtype=np.float32)
TABLE_ORIGIN_SHIFT_X = 0.615

CAM_T_BASE = np.array(
    [1.0795605013716922, -0.6623893074061149, 0.42449609765716245], dtype=np.float64
)
CAM_R_BASE_OPENCV = np.array(
    [
        [0.617938, -0.006099, -0.786203],
        [0.755988, -0.270051, 0.596284],
        [-0.215952, -0.962827, -0.162264],
    ],
    dtype=np.float64,
)
CAM_POSITION_JITTER_M = 0.10
CAM_ROTATION_JITTER_DEG = 2.0
CAM_CALIB_ERROR_TRANSLATION_STD_M = 0.0095284
CAM_CALIB_ERROR_ROTATION_STD_DEG = 1.4328

SIM_CAM_WIDTH = 164
SIM_CAM_HEIGHT = 96
CAM_VFOV_RAD: float = float(np.deg2rad(58.0))
CAM_NEAR, CAM_FAR = 0.4, 6.0


def _opencv_camera_rotation_to_sapien(r_opencv: np.ndarray) -> np.ndarray:
    """Convert an OpenCV/pinhole-optical camera rotation to SAPIEN's convention.

    Verbatim from master's ``reach_config``: OpenCV columns are
    [right, down, forward]; SAPIEN's are (forward, right, up) = (+x, -y, +z).
    Passing the raw OpenCV matrix into SAPIEN points the camera ~90 deg off.
    """
    right, down, forward = r_opencv[:, 0], r_opencv[:, 1], r_opencv[:, 2]
    return np.stack([forward, -right, -down], axis=1)


CAM_R_BASE = _opencv_camera_rotation_to_sapien(CAM_R_BASE_OPENCV)
CAM_Q_WXYZ = np.asarray(
    _Rotation.from_matrix(CAM_R_BASE).as_quat()[[3, 0, 1, 2]], dtype=np.float64
)

_ARM_BASE: type | None = None


def eval_arm_base() -> type:
    """Return the env class every eval env derives from, building it on first use.

    The class is defined inside this function so that importing the module does
    not pull in SAPIEN/ManiSkill (see the import note at the top). It is built
    once and cached.
    """
    global _ARM_BASE
    if _ARM_BASE is not None:
        return _ARM_BASE

    import sapien
    import torch
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils.structs.pose import Pose

    from mani_skill.utils import sapien_utils

    from pg3d.envs.xarm_adapter.reach_env import PG3DReachXArm7GripperEnv

    robot_base_pose = sapien.Pose(p=ROBOT_BASE_POSITION.tolist())

    class MasterSettingsXArm7GripperEnv(PG3DReachXArm7GripperEnv):
        """The local gripper reach env with master's camera and base/table layout."""

        # -- arm placement --------------------------------------------------
        def _load_agent(self, options: dict[str, Any]) -> None:
            BaseEnv._load_agent(self, options, robot_base_pose)

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
            super()._initialize_episode(env_idx, options)
            self._offset_table_for_origin_shift()
            # The local parent already jittered its own (wrong) nominal camera pose;
            # re-do it against master's nominal pose, which also re-samples this
            # episode's calibration error.
            self._randomize_camera_pose()

        def _offset_table_for_origin_shift(self) -> None:
            """Re-apply the table's origin-shift compensation, every episode.

            ManiSkill's TableSceneBuilder puts the table at a fixed ABSOLUTE world
            pose and resets it on every ``initialize()``, so this cannot be done
            once at load time. Verbatim behaviour of master's reach env.
            """
            table = getattr(getattr(self, "table_scene", None), "table", None)
            if table is None:
                return
            p = table.pose.p
            p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
            shifted = np.asarray(p, dtype=np.float32).reshape(-1, 3).copy()
            shifted[:, 0] += TABLE_ORIGIN_SHIFT_X
            table.set_pose(Pose.create_from_pq(p=shifted, q=table.pose.q))

        # -- cameras --------------------------------------------------------
        @property
        def _default_sensor_configs(self) -> list[CameraConfig]:
            cam_p = (ROBOT_BASE_POSITION.astype(np.float64) + CAM_T_BASE).tolist()
            pose = sapien.Pose(p=cam_p, q=CAM_Q_WXYZ.tolist())
            return [
                CameraConfig(
                    "base_camera",
                    pose,
                    SIM_CAM_WIDTH,
                    SIM_CAM_HEIGHT,
                    CAM_VFOV_RAD,
                    CAM_NEAR,
                    CAM_FAR,
                )
            ]

        @property
        def _default_human_render_camera_configs(self) -> CameraConfig:
            """master's third-person view: back from the reach-box centre.

            This repo's own render camera is still aimed at the M1 workspace
            (eye [0.7, 0, 0.45] -> target [-0.315, 0, 0.22]); with the base now
            at the origin that points back PAST the robot, away from the table,
            so videos show the arm from inches away against empty space.
            Confirmed by rendering Obs-PP-v1 before this override.
            """
            target = RENDER_CAMERA_TARGET
            eye = [target[0] + 1.0, target[1], target[2] + 0.25]
            pose = sapien_utils.look_at(eye=eye, target=list(target))
            return CameraConfig("render_camera", pose, 640, 480, float(np.deg2rad(60)), 0.1, 10.0)

        def _setup_sensors(self, options: dict | None = None) -> None:
            # Bypass this repo's gripper env, whose own _setup_sensors mounts an
            # extra `cam_wrist` sensor. master declares exactly one sensor
            # camera, so the mirror must too -- a second camera would change the
            # fused point cloud the policy is fed.
            BaseEnv._setup_sensors(self, options)

        def _randomize_camera_pose(self) -> None:
            """Jitter base_camera around MASTER's nominal pose, once per episode.

            Same magnitudes and the same structure as master's implementation:
            +/-10 cm and +/-2 deg uniform per axis on the rendering pose, plus an
            independent per-episode calibration-error correction that perturbs only
            the *belief* about where the camera was (applied in ``get_obs``), not
            the rendering pose.
            """
            camera = self._sensors.get("base_camera")
            if camera is None:
                self._camera_calib_correction = None
                return
            with torch.device(self.device):
                delta_p = (torch.rand(3) * 2.0 - 1.0) * CAM_POSITION_JITTER_M
                delta_euler_deg = (torch.rand(3) * 2.0 - 1.0) * CAM_ROTATION_JITTER_DEG
            nominal_p = ROBOT_BASE_POSITION.astype(np.float64) + CAM_T_BASE
            nominal_rot = _Rotation.from_quat(CAM_Q_WXYZ, scalar_first=True)
            delta_rot = _Rotation.from_euler("xyz", delta_euler_deg.cpu().numpy(), degrees=True)
            self._camera_episode_p = nominal_p + delta_p.cpu().numpy()
            self._camera_episode_rot = nominal_rot * delta_rot
            camera.camera.set_local_pose(
                sapien.Pose(
                    p=self._camera_episode_p.tolist(),
                    q=self._camera_episode_rot.as_quat(scalar_first=True).tolist(),
                )
            )
            self._camera_calib_correction = self._sample_camera_calibration_error()

    _ARM_BASE = MasterSettingsXArm7GripperEnv
    return _ARM_BASE


def gripper_open_qpos(agent: object) -> float:
    """Return the jaw-open drive target for this repo's gripper agent.

    This repo's ``XArm7Gripper`` exposes only ``_GRIPPER_CLOSED``; its mimic
    controller's lower bound is 0.0, and master backs its open target off that
    hard stop by 0.01 for the same PD/limit-solver reason the closed target is
    backed off. Use the same 0.01.
    """
    return float(getattr(agent, "_GRIPPER_OPEN", 0.01))


def repo_settings() -> dict[str, object]:
    """Describe this seam, for a run manifest."""
    return {
        "copy": "PG3D_Reach_Task (local testing mirror)",
        "arm_base_class": (
            "pg3d.envs.xarm_adapter.eval_envs.master_env_settings."
            "MasterSettingsXArm7GripperEnv"
        ),
        "camera": "master eye-on-base calibration, re-declared here; base_camera only",
        "camera_position": CAM_T_BASE.tolist(),
        "camera_render": [SIM_CAM_WIDTH, SIM_CAM_HEIGHT],
        "frame_shift": FRAME_SHIFT.tolist(),
        "crop_bounds": CROP_BOUNDS.tolist(),
        "gripper_control_mode": GRIPPER_CONTROL_MODE,
    }
