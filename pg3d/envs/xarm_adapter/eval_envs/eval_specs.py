"""eval_specs.py -- What an eval script reads off an eval env instead of the CLI.

The point of this suite is that a run needs an env id and nothing else: the
start pose, the policy goal, the object and obstacle layout, the goal-marker
contract and the frame budget are all frozen in the env. This module is the
read side of that promise:

``EvalEnvSpec``
    A frozen, JSON-serialisable description of one eval env.

:func:`eval_spec_for_env`
    Build it from a live env (also reachable as ``env.unwrapped.pg3d_eval_spec``).

:func:`env_episode_context`
    Return exactly the dict shape the eval scripts' ``--source dataset`` path
    produces from a Zarr episode (``state`` / ``target_position`` /
    ``tcp_pose`` / point-cloud fields). This is the adapter that lets a script
    drop ``--source dataset --episode-indices N`` and read the same fields from
    the env: no dataset episode, no CLI goal.

:func:`task_spec_for_env`
    Build a master :class:`~pg3d.tasks.spec.TaskSpec` for the env, so the same
    frozen scene can drive ``pg3d.tasks.live`` / ``PhaseExecutor`` without
    re-declaring any geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .eval_config import (
    GOAL_MARKER_POINTS,
    GOAL_MARKER_RADIUS,
    GOAL_MARKER_SHAPE,
    workspace_report,
)

Vec3 = tuple[float, float, float]


def _unwrap(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _tuple3(value: Any) -> Vec3:
    array = np.asarray(value, dtype=np.float32).reshape(-1)[:3]
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True)
class EvalEnvSpec:
    """Everything a run needs to know about one frozen eval env."""

    env_id: str
    variant: str
    task_ids: tuple[str, ...]

    #: TCP position the arm is at when the episode starts.
    start_tcp: Vec3
    #: Full robot qpos at the start (arm + gripper), as reset applies it.
    start_qpos: tuple[float, ...]
    #: Position the policy is conditioned on (reach goal, or place target).
    goal_position: Vec3
    #: Success radius the env grades its reach predicate with.
    goal_threshold_m: float

    #: Goal-marker contract the checkpoint was trained on.
    marker_points: int = GOAL_MARKER_POINTS
    marker_radius: float = GOAL_MARKER_RADIUS
    marker_shape: str = GOAL_MARKER_SHAPE

    #: Manipulation target ("cube" / a YCB model id), when the task has one.
    target_object: str | None = None
    target_object_position: Vec3 | None = None
    #: Obstacles and clutter, as ``{actor_name: position}``.
    obstacles: dict[str, Vec3] = field(default_factory=dict)
    clutter: dict[str, Vec3] = field(default_factory=dict)

    #: Env step cap declared by the registration.
    max_episode_steps: int | None = None
    #: Frame the positions are expressed in, plus the bounds audit.
    frame: dict[str, Any] = field(default_factory=workspace_report)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe copy, for a run manifest."""
        return asdict(self)


def eval_spec_for_env(env: Any) -> EvalEnvSpec:
    """Read one env's frozen spec off the live env."""
    unwrapped = _unwrap(env)
    env_id = str(getattr(getattr(unwrapped, "spec", None), "id", type(unwrapped).__name__))
    max_steps = getattr(getattr(unwrapped, "spec", None), "max_episode_steps", None)

    obstacles: dict[str, Vec3] = {}
    for index, actor in enumerate(getattr(unwrapped, "obstacle_actors", []) or []):
        obstacles[f"obstacle_{index}"] = _tuple3(_numpy(actor.pose.p).reshape(-1, 3)[0])

    clutter: dict[str, Vec3] = {}
    target_object: str | None = None
    target_object_position: Vec3 | None = None
    if getattr(unwrapped, "cube", None) is not None:
        target_object = "cube"
        target_object_position = _tuple3(unwrapped.CUBE_POS)
    elif getattr(unwrapped, "target_object", None) is not None:
        layout = type(unwrapped).layout()
        target_object = str(layout["target"]["model"])
        positions = unwrapped.object_positions()
        target_object_position = _tuple3(positions["target_object"])
        for index, item in enumerate(layout["clutter"]):
            clutter[str(item["model"])] = _tuple3(positions[f"clutter_{index}"])

    return EvalEnvSpec(
        env_id=env_id,
        variant=env_id.rsplit("-", 1)[-1],
        task_ids=tuple(getattr(unwrapped, "TASK_IDS", ())),
        start_tcp=_tuple3(unwrapped.START_TCP_POS),
        start_qpos=tuple(
            float(value) for value in _numpy(unwrapped.agent.robot.get_qpos()).reshape(-1)
        ),
        goal_position=_tuple3(unwrapped.GOAL_POS),
        goal_threshold_m=float(unwrapped.goal_thresh),
        target_object=target_object,
        target_object_position=target_object_position,
        obstacles=obstacles,
        clutter=clutter,
        max_episode_steps=int(max_steps) if max_steps else None,
    )


def env_episode_context(env: Any, obs: Any = None, info: Any = None) -> dict[str, Any]:
    """Return an env-sourced episode context shaped like the Zarr one.

    The eval scripts' ``--source dataset`` path builds this dict from row 0 of a
    Zarr episode and then uses it to (a) force the robot's start qpos, (b) place
    ``goal_site`` at ``target_position``, and (c) seed the first policy
    observation. An eval env already owns (a) and (b) and has reset itself to
    them, so this returns the same keys read back from the live env -- letting a
    script keep one code path while dropping ``--episode-indices``.

    ``point_cloud`` / ``robot_mask`` / ``point_valid_mask`` are filled in only
    when ``obs`` is passed, since they come from the render, not the spec. A
    script that seeds its own first observation from ``rollout_observation_entry``
    (the normal case) can ignore them.
    """
    unwrapped = _unwrap(env)
    qpos = _numpy(unwrapped.agent.robot.get_qpos()).reshape(-1).astype(np.float32)
    tcp_pose = _numpy(unwrapped.agent.tcp_pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)
    context: dict[str, Any] = {
        "episode_index": None,
        "episode_start": 0,
        "source": "eval_env",
        "env_id": eval_spec_for_env(unwrapped).env_id,
        "state": qpos,
        "target_position": np.asarray(unwrapped.GOAL_POS, dtype=np.float32).reshape(3),
        "tcp_pose": tcp_pose,
    }
    if obs is not None:
        from pg3d.envs.maniskill_adapter.observation import adapt_observation

        adapted = adapt_observation(obs, info=info, env=unwrapped)
        context["point_cloud"] = np.asarray(adapted.point_cloud, dtype=np.float32)
        context["robot_mask"] = np.asarray(adapted.robot_mask, dtype=bool)
        context["point_valid_mask"] = np.ones(context["point_cloud"].shape[0], dtype=bool)
    return context


def task_spec_for_env(env: Any, *, seed: int = 0, task_frames: int | None = None) -> Any:
    """Build a master ``TaskSpec`` from a frozen eval env.

    Reach envs get a single P1 position goal. Pick-and-place and cluttered envs
    get the P1--P4 phase graph with the target entity bound, but no
    ``candidate_set``: grasp candidates come from the analytic generator or
    GraspGen at eval time through their own port, and freezing one here would
    quietly override the method being compared.

    Obstacles become ``obstacle`` scene entities plus one whole-robot
    ``avoid_region`` binding each, with the geometry fingerprint that
    ``smoke_specs.verify_scene_constraint_match`` checks, so the constraint and
    the actor cannot disagree about the box they describe.
    """
    from pg3d.geometry import BoxRegion
    from pg3d.tasks.canonical import fingerprint
    from pg3d.tasks.pose import Pose3D
    from pg3d.tasks.spec import (
        ConstraintBinding,
        EpisodeBudget,
        MetricsSpec,
        Phase,
        PhaseSpec,
        PositionGoalSpec,
        SceneEntitySpec,
        SceneSpec,
        SeedBundle,
        TaskIdentity,
        TaskSpec,
    )

    from .eval_config import OBSTACLE_HALF_SIZES

    unwrapped = _unwrap(env)
    spec = eval_spec_for_env(unwrapped)
    is_manipulation = spec.target_object is not None

    entities: list[SceneEntitySpec] = []
    constraints: list[ConstraintBinding] = []
    active_phases = (
        (Phase.P1_REACH, Phase.P3_TRANSPORT) if is_manipulation else (Phase.P1_REACH,)
    )
    for name, position in spec.obstacles.items():
        region = BoxRegion(
            center=np.asarray(position, dtype=np.float32),
            half_extents=np.asarray(OBSTACLE_HALF_SIZES, dtype=np.float32),
        )
        digest = fingerprint(
            {
                "family": "box",
                "center": list(position),
                "half_extents": list(OBSTACLE_HALF_SIZES),
                "yaw": 0.0,
            }
        )
        entities.append(
            SceneEntitySpec(
                entity_id=name,
                role="obstacle",
                pose=Pose3D(position=position),
                geometry=region.to_json(),
                collidable=bool(getattr(unwrapped, "_obstacle_collision", True)),
                metadata={"geometry_fingerprint": digest},
            )
        )
        constraints.append(
            ConstraintBinding(
                binding_id=f"avoid_{name}",
                constraint={
                    "type": "avoid_region",
                    "target": "robot",
                    "region": region.to_json(),
                    "margin": 0.0,
                    "name": f"avoid_{name}",
                },
                active_phases=active_phases,
                required=True,
                metadata={"geometry_fingerprint": digest},
            )
        )

    if is_manipulation:
        entity_id = "cube" if spec.target_object == "cube" else "target_object"
        target_geometry = None
        if spec.target_object == "cube":
            half = float(unwrapped.CUBE_HALF_SIZE)
            target_geometry = BoxRegion(
                center=np.asarray(spec.target_object_position, dtype=np.float32),
                half_extents=np.asarray([half] * 3, dtype=np.float32),
            ).to_json()
        entities.append(
            SceneEntitySpec(
                entity_id=entity_id,
                role="target",
                pose=Pose3D(position=spec.target_object_position),
                geometry=target_geometry,
                asset_uri=None if spec.target_object == "cube" else f"ycb:{spec.target_object}",
            )
        )
        for model, position in spec.clutter.items():
            entities.append(
                SceneEntitySpec(
                    entity_id=f"clutter_{model}",
                    role="distractor",
                    pose=Pose3D(position=position),
                    asset_uri=f"ycb:{model}",
                )
            )
        phases = (
            PhaseSpec(Phase.P1_REACH, target_entity_id=entity_id),
            PhaseSpec(Phase.P2_GRASP, target_entity_id=entity_id),
            PhaseSpec(Phase.P3_TRANSPORT, goal_id="place_position", target_entity_id=entity_id),
            PhaseSpec(Phase.P4_RELEASE, goal_id="place_position", target_entity_id=entity_id),
        )
        goals = (
            PositionGoalSpec(
                goal_id="place_position",
                position=spec.goal_position,
                entity_id=entity_id,
            ),
        )
        metrics = MetricsSpec(
            task_success=("stable_object_position",),
            safety=("physical_collision",) if spec.obstacles else (),
            diagnostics=("grasp", "drop", "final_orientation"),
        )
        family = "pick_place"
    else:
        phases = (PhaseSpec(Phase.P1_REACH, goal_id="reach_position"),)
        goals = (
            PositionGoalSpec(
                goal_id="reach_position",
                position=spec.goal_position,
                translation_tolerance_m=spec.goal_threshold_m,
            ),
        )
        metrics = MetricsSpec(
            task_success=("stable_position_goal",),
            safety=("min_clearance",) if spec.obstacles else (),
            diagnostics=("terminal_rotation_error",),
        )
        family = "reach"

    budget = (
        EpisodeBudget(task_frames=int(task_frames))
        if task_frames is not None
        else EpisodeBudget(task_frames=int(spec.max_episode_steps or 150))
    )
    return TaskSpec(
        identity=TaskIdentity(
            task_id=f"{'/'.join(spec.task_ids) or 'eval'}:{spec.env_id}",
            family=family,
            instance_id=spec.variant,
        ),
        scene=SceneSpec(
            entities=tuple(entities),
            metadata={
                "source": "pg3d.envs.xarm_adapter.eval_envs",
                "env_id": spec.env_id,
                "start_tcp": list(spec.start_tcp),
                "frame": spec.frame,
            },
        ),
        goals=goals,
        phases=phases,
        constraints=tuple(constraints),
        seeds=SeedBundle(simulator=seed, policy=seed + 1),
        metrics=metrics,
        budget=budget,
        metadata={
            "goal_marker": {
                "num_points": spec.marker_points,
                "radius": spec.marker_radius,
                "shape": spec.marker_shape,
            }
        },
    )
