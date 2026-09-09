"""eval_base.py -- Shared base for every deterministic PG3D eval environment.

What this base guarantees, so no eval script has to supply it
-------------------------------------------------------------
1. Camera and arm come from the repo seam (``master_env_settings``), which in
   the master repo is the deployment reach env itself: the eye-on-base RealSense
   D455 calibration, the 164x96 aspect-matched render, the per-episode camera
   jitter and calibration-error model, the base-at-origin placement with the
   compensating table shift, and the ``xarm7_gripper`` agent. Nothing is
   re-declared here, so this suite cannot drift from the deployment env.
2. The arm starts AT this env's frozen start TCP, which the obstacle and object
   layouts were arranged around. The reach families (T1, T2) each have their
   OWN start; the manipulation families share the suite default. Reset applies a
   baked joint configuration -- no planner at runtime -- and then verifies the
   achieved TCP, raising if it misses by more than ``START_TCP_TOLERANCE_M``
   rather than running a mis-posed episode. An mplib IK fallback covers a start
   with no bake.
3. The policy goal is frozen per env: ``goal_half_extents`` is zero, so the
   inherited goal sampler can only produce ``GOAL_POS``. No ``--episode-indices``
   and no dataset episode is needed to place a goal.
4. The goal marker is the SPHERE contract the current checkpoint was trained on
   (192 points, 0.055 m shell). ``goal_marker_config`` states it and
   ``goal_marker_cloud`` renders it, so a script inserts the marker the
   checkpoint expects without hardcoding anything. Nothing in the scene draws
   the marker: it exists only in the point cloud a script appends it to.
5. The start and goal markers are STRICTLY VIRTUAL. They exist only for the
   human render camera (so a person can see where the arm must start and end)
   and contribute nothing to any observation: before every observation render
   each marker actor is hidden AND moved out of the scene, and restored right
   after. Nothing about them can reach the policy except through the goal-marker
   points a script appends deliberately (rule 4). ``marker_visibility="off"``
   removes their render bodies outright for a paranoid run.

   Obstacle bars, the cube and the YCB objects are NOT markers: they are real
   scene geometry the policy is meant to see, and are never hidden.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose

from pg3d.policies.dp3.goal_markers import goal_marker_points, insert_goal_marker_points

from .eval_config import (
    GOAL_MARKER_POINTS,
    GOAL_MARKER_RADIUS,
    GOAL_MARKER_SHAPE,
    START_TCP,
    SUITE_START_QPOS,
    START_TCP_TOLERANCE_M,
    verify_marker_contract,
    workspace_box_edges,
)
from .master_env_settings import eval_arm_base, gripper_open_qpos

Vec3 = tuple[float, float, float]

# Rest-pose TCP orientation (wxyz): 180 deg about x, so the tool z-axis points
# down at the table. Every reach goal in this project is planned to it, so it is
# also the orientation the start-pose IK fallback solves for.
_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


#: Where a marker actor is parked while observations render: far below the
#: table, metres outside the point-cloud crop box and outside every camera
#: frustum. Used as the hard guarantee that a marker cannot contribute a depth
#: pixel even if a renderer ignores visibility.
MARKER_STASH_POSITION: Vec3 = (0.0, 0.0, -50.0)


def _set_actor_visibility(actor: Any, visibility: float) -> None:
    """Set one actor's render-body visibility without touching its physics.

    Tolerates actors whose render component exposes only ``set_visible``, and
    actors with no render body at all (``marker_visibility="off"``).
    """
    try:
        import sapien.render as sr
    except ImportError:  # pragma: no cover - sapien always ships render in sim
        return
    for obj in getattr(actor, "_objs", [actor]):
        body = obj.find_component_by_type(sr.RenderBodyComponent)
        if body is None:
            continue
        if hasattr(body, "set_visibility"):
            body.set_visibility(visibility)
        elif hasattr(body, "set_visible"):
            body.set_visible(visibility > 0.5)


def _remove_render_bodies(actor: Any) -> None:
    """Strip an actor's render bodies entirely: no visual, in any camera.

    This is what ``marker_visibility="off"`` uses. The actor stays in the scene
    as a pose carrier (the env still writes the goal/start into it, and
    ``pg3d_eval_spec`` still reads it), it simply has nothing to draw.
    """
    try:
        import sapien.render as sr
    except ImportError:  # pragma: no cover
        return
    for obj in getattr(actor, "_objs", [actor]):
        body = obj.find_component_by_type(sr.RenderBodyComponent)
        if body is None:
            continue
        try:
            obj.remove_component(body)
        except Exception:
            # Older SAPIEN builds expose no component removal; a permanently
            # invisible body is equivalent for every camera.
            _set_actor_visibility(obj, 0.0)


def _clone_raw_pose(raw_pose: Any) -> Any:
    """Detached copy of a ``(N, 7)`` raw pose, whether it is torch or numpy."""
    clone = getattr(raw_pose, "clone", None)
    if clone is not None:
        return clone()
    return torch.as_tensor(np.array(raw_pose, copy=True), dtype=torch.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


class PG3DEvalBase(eval_arm_base()):  # type: ignore[misc]
    """Deterministic eval base: frozen start, frozen goal, virtual markers.

    Subclasses set ``GOAL_POS`` (the position the policy is conditioned on) and
    may set ``START_TCP_POS`` when a variant needs a start other than the
    suite-wide one. Everything else -- scene, camera, agent, marker contract --
    is inherited.
    """

    #: World-frame position the policy is steered to. Reach variants use their
    #: reach goal; pick-and-place variants use their place target.
    GOAL_POS: Vec3

    #: World-frame TCP position the arm must be at when the episode starts.
    START_TCP_POS: Vec3 = START_TCP

    #: Arm joint configuration (joint1..joint7) that reaches ``START_TCP_POS``,
    #: baked in eval_config so no planner runs at reset. The achieved TCP is
    #: still verified against ``START_TCP_POS`` every episode.
    START_QPOS: tuple[float, ...] = SUITE_START_QPOS

    #: ``"closed"`` keeps the reach-era hold; ``"open"`` opens the jaws at reset
    #: so a pick phase can close them onto an object.
    GRIPPER_START: str = "closed"

    #: Human-readable task ids from docs/evaluation_tasks_scratchpad.md that
    #: this env instance serves (the same env backs several tasks; the task
    #: difference lives in the constraints an eval script binds, not the scene).
    TASK_IDS: tuple[str, ...] = ()

    def __init__(
        self,
        *args: Any,
        show_workspace: bool = False,
        strict_start: bool = True,
        allow_goal_override: bool = False,
        marker_visibility: str = "viz",
        **kwargs: Any,
    ) -> None:
        if marker_visibility not in ("viz", "off"):
            raise ValueError(
                f"marker_visibility must be 'viz' or 'off', got {marker_visibility!r}"
            )
        self._marker_visibility = marker_visibility
        self._show_workspace = bool(show_workspace)
        self._strict_start = bool(strict_start)
        self._workspace_actors: list[Any] = []
        self._start_qpos_cache: np.ndarray | None = None
        # goal_site's true position, saved while the markers are stashed for a
        # render (see _hide_markers). Non-None ONLY inside get_obs.
        self._goal_p_while_stashed: Any = None
        # Fails closed if the policy package's marker defaults ever move away
        # from the contract this suite (and the checkpoint's bake) assume.
        verify_marker_contract()
        # A zero-extent goal region makes the inherited sampler deterministic:
        # the only position it can draw is this env's frozen goal.
        #
        # These are FORCED, not defaulted. Eval scripts build env kwargs from a
        # training dataset's recorded `env_kwargs`, which carries that dataset's
        # `goal_center` / `goal_half_extents` -- honouring them would silently
        # restore random goal sampling and quietly un-freeze the whole suite.
        # Pass allow_goal_override=True for a deliberate sensitivity study.
        frozen_goal = tuple(float(value) for value in self.GOAL_POS)
        if not allow_goal_override:
            requested_center = kwargs.get("goal_center")
            requested_extents = kwargs.get("goal_half_extents")
            overridden = [
                name
                for name, requested, forced in (
                    ("goal_center", requested_center, frozen_goal),
                    ("goal_half_extents", requested_extents, (0.0, 0.0, 0.0)),
                )
                if requested is not None
                and tuple(float(v) for v in requested) != tuple(float(v) for v in forced)
            ]
            if overridden:
                warnings.warn(
                    f"[{type(self).__name__}] ignoring caller-supplied {', '.join(overridden)}: "
                    "this eval env's goal is frozen. Pass allow_goal_override=True if you "
                    "really mean to sample goals here.",
                    stacklevel=2,
                )
            kwargs["goal_center"] = frozen_goal
            kwargs["goal_half_extents"] = (0.0, 0.0, 0.0)
        else:
            kwargs.setdefault("goal_center", frozen_goal)
            kwargs.setdefault("goal_half_extents", (0.0, 0.0, 0.0))
        # Goal regions would re-introduce sampling behind the goal_center check.
        if kwargs.get("goal_regions") and not allow_goal_override:
            warnings.warn(
                f"[{type(self).__name__}] dropping caller-supplied goal_regions: "
                "this eval env's goal is frozen.",
                stacklevel=2,
            )
            kwargs["goal_regions"] = ()
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)

        if self._show_workspace:
            self._workspace_actors = []
            for index, edge in enumerate(workspace_box_edges()):
                self._workspace_actors.append(
                    actors.build_box(
                        self.scene,
                        half_sizes=list(edge["half_sizes"]),
                        color=[0.0, 1.0, 0.0, 0.8],
                        name=f"ws_edge_{index}",
                        body_type="kinematic",
                        add_collision=False,
                        initial_pose=sapien.Pose(p=list(edge["centre"])),
                    )
                )

        if self._marker_visibility == "off":
            # Strictly virtual with no visual at all: the markers keep carrying
            # the start/goal poses (the spec and any grader still read them) but
            # nothing draws them, in any camera.
            for actor in self._marker_actors():
                _remove_render_bodies(actor)

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        # Parent chain: rest keyframe -> table -> goal sample -> table shift ->
        # camera jitter + calibration error.
        super()._initialize_episode(env_idx, options)
        batch = len(env_idx)
        with torch.device(self.device):
            self._apply_start_pose(batch)
            self._place_goal(batch)

    def _place_goal(self, batch: int) -> None:
        """Pin goal_site to this env's frozen goal."""
        goal = torch.tensor(
            [list(self.GOAL_POS)], dtype=torch.float32, device=self.device
        ).expand(batch, -1)
        self.goal_site.set_pose(Pose.create_from_pq(goal))

    def _apply_start_pose(self, batch: int) -> None:
        """Put the arm at ``START_TCP_POS`` and record it on ``start_site``.

        The rest keyframe's FK is the frozen start, so the common path only sets
        the gripper width and verifies. IK runs solely as a fallback for a
        variant that overrides ``START_TCP_POS``, and a start that cannot be
        achieved raises rather than silently shifting the layout the obstacles
        and objects were arranged around.
        """
        qpos = np.asarray(self.agent.keyframes["rest"].qpos, dtype=np.float32).copy()
        if self.START_QPOS is not None:
            arm = np.asarray(self.START_QPOS, dtype=np.float32).reshape(-1)
            qpos[: arm.size] = arm
        if self.GRIPPER_START == "open":
            qpos[7:] = gripper_open_qpos(self.agent)
        elif self.GRIPPER_START != "closed":
            raise ValueError(
                f"GRIPPER_START must be 'closed' or 'open', got {self.GRIPPER_START!r}"
            )

        self._reset_to_qpos(qpos, batch)
        error = self._start_tcp_error()
        if error > START_TCP_TOLERANCE_M:
            solved = self._solve_start_qpos(qpos)
            if solved is not None:
                self._reset_to_qpos(solved, batch)
                error = self._start_tcp_error()
        if error > START_TCP_TOLERANCE_M:
            message = (
                f"[{type(self).__name__}] arm start TCP is {error:.4f} m from the frozen "
                f"START_TCP_POS {tuple(round(v, 4) for v in self.START_TCP_POS)} "
                f"(tolerance {START_TCP_TOLERANCE_M} m). The obstacle and object layouts "
                "were arranged around that start, so this episode's clearances are not the "
                "frozen ones."
            )
            if self._strict_start:
                raise RuntimeError(message)
            warnings.warn(message, stacklevel=2)

        start = torch.tensor(
            [list(self.START_TCP_POS)], dtype=torch.float32, device=self.device
        ).expand(batch, -1)
        self.start_site.set_pose(Pose.create_from_pq(start))

    def _reset_to_qpos(self, qpos: np.ndarray, batch: int) -> None:
        tensor = (
            torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
            .reshape(1, -1)
            .expand(batch, -1)
        )
        self.agent.reset(tensor)

    def _start_tcp_error(self) -> float:
        """Distance between the achieved TCP and the frozen start, in metres."""
        achieved = _to_numpy(self.agent.tcp_pose.p).reshape(-1, 3)[0].astype(np.float64)
        target = np.asarray(self.START_TCP_POS, dtype=np.float64)
        return float(np.linalg.norm(achieved - target))

    def _solve_start_qpos(self, seed_qpos: np.ndarray) -> np.ndarray | None:
        """Return a qpos whose FK reaches ``START_TCP_POS``, or None.

        Only reached when a subclass overrides ``START_TCP_POS`` away from the
        rest keyframe's FK. mplib is imported here so importing this module
        never pulls in the planner.
        """
        if self._start_qpos_cache is not None:
            return self._start_qpos_cache
        try:
            from pg3d.envs.xarm_adapter.motionplanner import XArm7GripperMotionPlanningSolver

            solver = XArm7GripperMotionPlanningSolver(
                self,
                debug=False,
                vis=False,
                base_pose=self.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
            planner = solver.planner
            n_ik = len(planner.user_joint_names)
            seed = np.asarray(seed_qpos, dtype=np.float64).reshape(-1)[:n_ik]
            goal = np.hstack(
                [np.asarray(self.START_TCP_POS, dtype=np.float64), _DOWN_QUAT_WXYZ]
            )
            status, solutions = planner.IK(goal, seed, n_init_qpos=20, threshold=1e-3)
        except Exception as exc:  # pragma: no cover - depends on mplib at runtime
            warnings.warn(
                f"[{type(self).__name__}] start-pose IK unavailable "
                f"({type(exc).__name__}: {exc}); keeping the rest keyframe.",
                stacklevel=2,
            )
            return None
        if status != "Success" or solutions is None or len(solutions) == 0:
            return None
        solved = np.asarray(seed_qpos, dtype=np.float32).copy()
        arm = np.asarray(solutions[0], dtype=np.float32).reshape(-1)
        solved[: arm.size] = arm
        self._start_qpos_cache = solved
        return solved

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _marker_actors(self) -> list[Any]:
        """Every strictly-virtual actor: visualisation only, never observed.

        Deliberately excludes the cube, the YCB objects and the obstacle bars --
        those are real scene geometry the policy must see.
        """
        markers = [
            getattr(self, "goal_site", None),
            getattr(self, "start_site", None),
        ]
        markers.extend(self._workspace_actors)
        return [actor for actor in markers if actor is not None]

    def _hide_markers(self) -> list[tuple[Any, Any]]:
        """Take every marker out of the scene for the duration of a render.

        Two mechanisms, because one alone is not a guarantee:

        * ``Actor.hide_visual()`` is ManiSkill's own, and is backend-correct --
          on CPU it zeroes render-body visibility, on GPU (where visibility is
          not respected) it teleports the body away and flushes the GPU pose
          buffer.
        * On CPU we ALSO park the actor at ``MARKER_STASH_POSITION``, far
          outside every camera frustum and the crop box, so a marker cannot
          contribute a depth pixel even if a renderer draws a zero-visibility
          body into the depth buffer. On GPU this is skipped: ``hide_visual``
          already moved the body, and an unflushed pose write there would be a
          silent no-op that only muddies the guarantee.

        Returns the saved poses, for :meth:`_show_markers`.
        """
        import sapien.physx as physx

        gpu = bool(physx.is_gpu_enabled())
        # Both stash mechanisms (the CPU park at MARKER_STASH_POSITION and
        # hide_visual's GPU teleport) move goal_site's pose, so anything that
        # reads that pose during the render -- _get_obs_extra's `goal_pos`,
        # which is the ONLY channel the goal reaches the policy through -- would
        # read the stash instead of the goal. Remember the real one first.
        self._goal_p_while_stashed = _clone_raw_pose(self.goal_site.pose.p)
        saved: list[tuple[Any, Any]] = []
        for actor in self._marker_actors():
            pose = None if gpu else _clone_raw_pose(actor.pose.raw_pose)
            saved.append((actor, pose))
            if not gpu:
                stash = torch.tensor(
                    [list(MARKER_STASH_POSITION)], dtype=torch.float32, device=self.device
                ).expand(pose.shape[0], -1)
                actor.set_pose(Pose.create_from_pq(stash))
            hide = getattr(actor, "hide_visual", None)
            if hide is not None:
                hide()
            else:
                _set_actor_visibility(actor, 0.0)
        return saved

    def _show_markers(self, saved: list[tuple[Any, Any]]) -> None:
        """Undo :meth:`_hide_markers`, restoring poses last so they always win."""
        self._goal_p_while_stashed = None
        for actor, pose in saved:
            show = getattr(actor, "show_visual", None)
            if show is not None:
                show()
            elif self._marker_visibility == "viz":
                _set_actor_visibility(actor, 1.0)
            if pose is not None:
                actor.set_pose(Pose.create_from_pq(p=pose[..., :3], q=pose[..., 3:]))
        if self._marker_visibility == "off":
            # `show_visual` would undo the render-body strip on backends where
            # removal fell back to permanent invisibility.
            for actor, _ in saved:
                _set_actor_visibility(actor, 0.0)

    def goal_position_p(self) -> Any:
        """goal_site's true ``(N, 3)`` position, stash-proof.

        Inside :meth:`get_obs` the marker actors are parked out of the scene, so
        ``self.goal_site.pose.p`` is the stash position, not the goal. Every
        read of the goal that can run during a render must go through here.
        """
        if self._goal_p_while_stashed is not None:
            return self._goal_p_while_stashed
        return self.goal_site.pose.p

    def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        """Parent extras with the goal read through the stash-proof accessor.

        ``goal_pos`` is what the observation adapter turns into the policy's
        ``goal_xyz`` and its point-cloud goal marker, so the parent's raw
        ``goal_site.pose.p`` read would condition the policy on
        ``MARKER_STASH_POSITION`` (0, 0, -50) for the entire episode.
        """
        extra = super()._get_obs_extra(info)
        goal_p = self.goal_position_p()
        extra["goal_pos"] = goal_p
        extra["tcp_to_goal_pos"] = goal_p - self.agent.tcp_pose.p
        return extra

    def get_obs(self, info: dict[str, Any] | None = None, unflattened: bool = False) -> Any:
        """Render observations with every marker actor out of the scene.

        The markers must stay in the scene between renders (the video and the
        human render camera read their poses, which is their whole purpose) but
        must never reach an observation: the policy learns the goal only from
        the marker points a script appends to the point cloud, so a physical
        green ball in the cloud would be a second, unlearned goal cue.
        """
        saved = self._hide_markers()
        try:
            return super().get_obs(info, unflattened=unflattened)
        finally:
            self._show_markers(saved)

    # ------------------------------------------------------------------
    # Self-description consumed by eval scripts
    # ------------------------------------------------------------------
    @property
    def goal_marker_config(self) -> dict[str, Any]:
        """The marker contract this env's checkpoint was trained on."""
        return {
            "num_points": GOAL_MARKER_POINTS,
            "radius": GOAL_MARKER_RADIUS,
            "shape": GOAL_MARKER_SHAPE,
        }

    def goal_marker_cloud(self, target_position: Any = None) -> np.ndarray:
        """Return the ``(192, 3)`` marker points for ``target_position``.

        Defaults to this env's frozen goal, so a script can bake the marker
        without knowing the goal or the marker geometry.
        """
        target = self.GOAL_POS if target_position is None else target_position
        return goal_marker_points(
            np.asarray(target, dtype=np.float32),
            num_points=GOAL_MARKER_POINTS,
            radius=GOAL_MARKER_RADIUS,
            shape=GOAL_MARKER_SHAPE,
        )

    def insert_goal_marker(self, point_cloud: Any, target_position: Any = None) -> np.ndarray:
        """Overwrite the last 192 point-cloud slots with this env's goal marker.

        This is the ONLY channel through which the goal reaches the policy. Call
        it on the cropped/downsampled cloud that is about to be fed to the
        policy, exactly as the dataset writer baked it -- the env's own contract
        (192 points, 0.055 m, sphere) is applied, so a script cannot pass the
        wrong marker by accident.
        """
        target = self.GOAL_POS if target_position is None else target_position
        return insert_goal_marker_points(
            point_cloud,
            np.asarray(target, dtype=np.float32),
            num_points=GOAL_MARKER_POINTS,
            radius=GOAL_MARKER_RADIUS,
            shape=GOAL_MARKER_SHAPE,
        )

    def _marker_ball_point_count(self, obs: Any, tolerance_m: float) -> int:
        """Points inside any marker's ball, in one unflattened observation."""
        cloud = obs.get("pointcloud") if isinstance(obs, dict) else None
        if cloud is None:
            raise ValueError("marker leak checks need an unflattened pointcloud observation")
        xyzw = _to_numpy(cloud["xyzw"]).reshape(-1, 4)
        points = xyzw[xyzw[:, 3] > 0][:, :3]
        total = 0
        for centre, radius in (
            (np.asarray(self.GOAL_POS, dtype=np.float32), float(self.goal_thresh)),
            (np.asarray(self.START_TCP_POS, dtype=np.float32), float(self.goal_thresh)),
        ):
            distance = np.linalg.norm(points - centre.reshape(1, 3), axis=1)
            total += int(np.count_nonzero(distance <= radius + tolerance_m))
        return total

    def set_marker_visibility(self, visible: bool) -> None:
        """Show/hide the marker actors for the human viewer, live.

        Only affects what a person sees between observation renders: every
        observation re-hides them regardless (:meth:`get_obs`), so this cannot
        leak a marker into the policy's input. Used by the interactive
        inspector's marker toggle.
        """
        self._marker_visibility = "viz" if visible else "off"
        for actor in self._marker_actors():
            _set_actor_visibility(actor, 1.0 if visible else 0.0)

    def marker_leak_report(self, *, tolerance_m: float = 0.005) -> dict[str, Any]:
        """Prove the markers are excluded from observations, differentially.

        Counting points near a marker in a single observation cannot answer this:
        a place target sits 3.5 cm above the table, so the table itself puts
        plenty of legitimate points inside a 5.5 cm ball around it. The test
        that *is* decisive renders the same state twice -- once through
        :meth:`get_obs` (markers hidden, the real path) and once through the
        parent directly (markers left in place) -- and compares:

        * ``hidden`` counts points inside the marker balls on the real path;
          they are all ambient geometry (table, cube, obstacle bars);
        * ``visible`` counts them with the markers present;
        * ``removed = visible - hidden`` is the marker's own contribution.

        ``removed > 0`` proves the hiding actually removes marker points from
        the observation. ``removed == 0`` means the hiding did nothing -- either
        the markers never rendered (fine, but unproven) or they are leaking.
        """
        info = self.get_info()
        hidden = self._marker_ball_point_count(
            self.get_obs(info, unflattened=True), tolerance_m
        )
        visible = self._marker_ball_point_count(
            super().get_obs(info, unflattened=True), tolerance_m
        )
        return {
            "hidden": hidden,
            "visible": visible,
            "removed": visible - hidden,
            "hiding_effective": visible - hidden > 0,
            "tolerance_m": tolerance_m,
        }

    def assert_markers_excluded(self, *, tolerance_m: float = 0.005) -> dict[str, Any]:
        """Raise unless hiding demonstrably removes the markers from the cloud."""
        report = self.marker_leak_report(tolerance_m=tolerance_m)
        if not report["hiding_effective"]:
            raise AssertionError(
                f"[{type(self).__name__}] marker hiding removed no points from the "
                f"observation ({report}); the start/goal markers may be leaking into "
                "the point cloud."
            )
        return report

    @property
    def pg3d_eval_spec(self) -> Any:
        """This env's frozen :class:`EvalEnvSpec` (start, goal, scene, budget)."""
        from .eval_specs import eval_spec_for_env

        return eval_spec_for_env(self)
