# Pose-reach data generation — full-coverage run

Full-coverage generation command for the xArm7-with-gripper 6D pose-reach dataset.
Maximizes the variation axes (approach direction × in-place roll × null-space arm
configs × trajectory-path families) so the dataset broadly prepares a DP3 policy
for pick-and-place reaching.

`write_xarm7_pose_reach_dataset.py` is a thin wrapper that injects xArm7 workspace
bounds / gripper defaults and calls `write_maniskill_pose_reach_dataset.py::main`,
so every flag below is a flag of the underlying writer.

## Command (copy-paste this one)

```bash
uv run python scripts/write_xarm7_pose_reach_dataset.py \
  --num-demos 6000 \
  --max-attempts 3000 \
  --num-workers 8 \
  --goal-marker-style triad \
  --orientation-cone-half-angle-deg 60 \
  --orientations-per-goal 25 \
  --configs-per-goal-pose 3 \
  --config-match-position-tol-m 0.01 \
  --config-match-orientation-tol-deg 3.0 \
  --config-min-joint-separation-rad 0.35 \
  --max-configs-per-start-goal 60 \
  --trajectory-variants-per-reset 12 \
  --min-feasible-families 4 \
  --waypoint-attempts 80 \
  --smooth-trajectory \
  --curved-paths --curvature-std 0.10 \
  --hold-steps 8 \
  --num-points 1024 \
  --output artifacts/reach-datasets/pg3d-xarm7-pnp-cover.zarr --overwrite
```

## Command (annotated reference — do NOT copy-paste; inline `#` after `\` breaks bash)

Each line: `--flag value   # what it means / how to tune`.

```bash
uv run python scripts/write_xarm7_pose_reach_dataset.py \
  --num-demos 6000 \                       # total episodes SAVED (hard cap). Raise for a bigger dataset; the cross-product per seed is large, so this is the main size knob.
  --max-attempts 3000 \                    # max env resets/seeds tried to hit --num-demos. Raise if many seeds get rejected (infeasible goals); lower to fail fast.
  --num-workers 8 \                        # Ray parallel workers (each owns a sim env, num_gpus=0.05). Set 0 for sequential/debug. Lower if GPU OOM.
  --goal-marker-style triad \              # 'triad' bakes the oriented 6D goal frame (position+orientation); 'sphere' is legacy position-only. Keep 'triad' for pose reaching.
  --orientation-cone-half-angle-deg 60 \   # half-angle of the down-facing goal cone (60 = ~120 deg solid angle). Lower to restrict goals nearer straight-down; raise for more tilt.
  --orientations-per-goal 25 \             # equal-area goal orientations sampled per start/goal (approach dir + in-place roll folded in). Raise for finer orientation coverage; each multiplies episodes.
  --configs-per-goal-pose 3 \              # distinct null-space arm configs (j1..j7) that reach the SAME triad. Raise for more elbow/shoulder diversity per pose; feasibility limits how many exist.
  --config-match-position-tol-m 0.01 \     # max TCP position error (m) for a null-space config to count as reaching the goal. Tighten for stricter pose accuracy; loosen if too many rejects.
  --config-match-orientation-tol-deg 3.0 \ # max TCP orientation error (deg) to count as reaching the goal. Also what rejects the j7 0/180 flip. Keep small (<~5) so configs truly hit the triad.
  --config-min-joint-separation-rad 0.35 \ # min joint-space L2 distance between accepted configs (dedup). Raise to force more distinct configs; lower to keep near-duplicates.
  --max-configs-per-start-goal 60 \        # cap on total configs per start/goal across all orientations (bounds dataset size). Raise with --num-demos; lower to shrink per-seed blowup.
  --trajectory-variants-per-reset 12 \     # asymmetric trajectory-path families (left_wide, upper_arc, ...) per config = PATH multimodality. Lower for fewer routes per pose; 12 is the full set.
  --min-feasible-families 4 \              # min families that must plan successfully to keep a seed. Raise for stricter path diversity per seed; lower to keep more (partial) seeds.
  --waypoint-attempts 80 \                 # waypoint samples tried per family before giving up. Raise if families fail often in a tight workspace; lower to speed up.
  --smooth-trajectory \                    # arc-length re-time multi-segment plans to constant joint speed (removes seam deceleration). Pass --no-smooth-trajectory to ablate.
  --curved-paths \                         # random per-episode curved shapes (unique curve each episode) instead of fixed family ratios. Drop for straighter, more repeatable paths.
  --curvature-std 0.10 \                   # std of the Gaussian curvature (family-scale units) when --curved-paths. Raise for more curl; keep modest to avoid mode-averaging.
  --hold-steps 8 \                         # terminal zero-velocity hold frames after success (teaches clean stop). 3-4 is enough; 8 is a safe default.
  --num-points 1024 \                      # points per saved point cloud (includes the trailing goal-marker slots). Match your policy encoder input size.
  --output artifacts/reach-datasets/pg3d-xarm7-pnp-cover.zarr --overwrite
```

## Module-level knobs (NO CLI flag — edit in `scripts/write_maniskill_pose_reach_dataset.py`)

```python
GRASP_ROLL_RANGE_DEG = 180.0             # in-place gripper roll psi ~ U[0, this) about the goal approach axis.
                                         # 180 = every distinct 2-finger parallel-jaw grasp angle once (no 180 duplicate).
                                         # Set 0.0 to disable roll (single wrist orientation per approach).
                                         # Folded into --orientations-per-goal (no separate roll-count flag).

START_APPROACH_CONE_HALF_ANGLE_DEG = 30.0  # episodes START with the gripper within this many deg of straight-down.
                                           # Widened 10 -> 30 so the policy sees more start-orientation variety
                                           # (reduces distribution shift when a place segment starts from a tilted
                                           # post-grasp pose). Raise further / bypass for pick-and-place place segments.
```

## What each variation axis gives you (link_tcp / triad framing)

- **Approach direction** (`--orientation-cone-half-angle-deg`, `--orientations-per-goal`):
  where the gripper points, filling the down-cone equal-area.
- **In-place roll** (`GRASP_ROLL_RANGE_DEG`): spins the finger line about the approach
  axis — covers "banana ∥x vs ∥y" grasp angles. Folded into `--orientations-per-goal`.
- **Null-space configs** (`--configs-per-goal-pose` + tolerances): multiple j1..j7 that
  land `link_tcp` on the SAME triad B; the j7 0/180 flip is rejected by the orientation
  tolerance, so it never appears as a duplicate solution.
- **Trajectory-path families** (`--trajectory-variants-per-reset`, `--curved-paths`):
  different ROUTES to the same pose (path multimodality).

## Pilot first

Confirm success rates before the full run:

```bash
uv run python scripts/write_xarm7_pose_reach_dataset.py \
  --num-demos 200 --max-attempts 300 --num-workers 4 \
  --goal-marker-style triad --orientations-per-goal 6 --configs-per-goal-pose 2 \
  --output artifacts/reach-datasets/pg3d-xarm7-pnp-pilot.zarr --overwrite
```

## Inspect what landed

```bash
uv run python scripts/diagnose_reach_dataset.py \
  --dataset artifacts/reach-datasets/pg3d-xarm7-pnp-cover.zarr
```

Check `orientation_coverage` (tilt min/mean/max, equal-area cap-cell fill),
`pose_multimodality` (configs-per-pose), and start-pose tilt in the episode metadata.

## Pick-and-place note

The start down-cone is a data-generation *sampling* constraint (applied only when the
writer randomizes a fresh start), NOT a runtime kinematic limit. For a true
pick-and-place "place" segment the start should be the pick's terminal (possibly
tilted) pose; widen/bypass `START_APPROACH_CONE_HALF_ANGLE_DEG` for that segment so
training and deployment start-distributions match. See
`docs/adr/0010-6d-goal-marker-and-pose-multimodality.md`.

Start-pose sampling in one line: the start gripper approach axis is sampled
equal-area with tilt **theta in [0, 30] deg** from straight-down and azimuth
**phi in [0, 360) deg**, roll = 0 (i.e. uniformly over the 30-deg down-cone).

---

## Training (GPU)

`train_dp3_reach.py` reads `goal_marker_points` / `goal_marker_radius` /
`goal_marker_style` from the dataset's `metadata.json`, so the policy is trained
(and later deployed) with the exact marker the dataset baked. No need to pass the
marker flags unless deliberately overriding.

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-datasets/pg3d-xarm7-pnp-cover.zarr \
  --device cuda \
  --max-steps 200000 \
  --batch-size 64 \
  --num-workers 8 \
  --val-ratio 0.1 --val-every 1000 \
  --lr 1e-4 --warmup-steps 1000 --grad-clip-norm 1.0 --use-ema \
  --wandb-mode online --wandb-project pg3d --wandb-name xarm7-pnp-cover \
  --checkpoint-dir artifacts/reach-datasets/xarm7-pnp-cover-ckpt \
  --checkpoint-every 10000
```

Annotated (reference — do NOT copy-paste, `#` after `\` breaks bash):

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset .../pg3d-xarm7-pnp-cover.zarr \  # the zarr from the data-gen command above
  --device cuda \                            # 'cuda' to train on GPU; 'cpu' for smoke only
  --max-steps 200000 \                       # total optimizer steps (NOT epochs). Raise for longer training
  --batch-size 64 \                          # sequences per step. Lower if GPU OOM
  --num-workers 8 \                          # dataloader workers. 0 for debug
  --val-ratio 0.1 --val-every 1000 \         # hold out 10% episodes for val; eval every 1000 steps
  --lr 1e-4 --warmup-steps 1000 \            # AdamW LR + linear warmup steps
  --grad-clip-norm 1.0 --use-ema \           # clip grads; keep an EMA copy (used by default at rollout)
  --wandb-mode online \                      # 'disabled' for no logging; 'offline' to sync later
  --wandb-project pg3d --wandb-name xarm7-pnp-cover \
  --checkpoint-dir .../xarm7-pnp-cover-ckpt \# where step_*.pt + final_step_*.pt land
  --checkpoint-every 10000                   # save a checkpoint every N steps
```

CPU smoke (2-epoch-ish functional check):

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset <smoke.zarr> --device cuda --max-steps 60 --batch-size 8 \
  --num-workers 0 --val-ratio 0.2 --val-every 30 --wandb-mode disabled \
  --checkpoint-dir artifacts/pose-reach-smoke/smoke-ckpt \
  --checkpoint-every 60 --no-checkpoint-rollout-videos
```

## Inference / rollout (writes videos + Rerun by default)

`rollout_dp3_reach_policy.py` runs the trained checkpoint live in ManiSkill and
writes, per episode, `episode_*.mp4` (video), `episode_*.rrd` (Rerun timeline),
plus `summary.json` and `metrics.jsonl`, into `--output-dir`. EMA weights are used
by default. The baked goal marker style is read from the checkpoint (triad), so
inference matches training.

Dataset-seed rollout (replays start/goal pairs from the dataset):

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --checkpoint artifacts/reach-datasets/xarm7-pnp-cover-ckpt/final_step_00200000.pt \
  --dataset artifacts/reach-datasets/pg3d-xarm7-pnp-cover.zarr \
  --source dataset --episodes 10 --device cuda \
  --output-dir artifacts/reach-datasets/xarm7-pnp-cover-rollouts-dataset
```

Fresh-seed rollout (new random start/goal pairs; xArm7 env):

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --checkpoint artifacts/reach-datasets/xarm7-pnp-cover-ckpt/final_step_00200000.pt \
  --dataset artifacts/reach-datasets/pg3d-xarm7-pnp-cover.zarr \
  --env-id-override PG3DReach-XArm7-Gripper-Workspace-v0 \
  --source fresh --episodes 10 --seed-start 20000 --device cuda \
  --output-dir artifacts/reach-datasets/xarm7-pnp-cover-rollouts-fresh
```

Annotated rollout flags (reference):

```bash
  --checkpoint <path.pt>            # trained checkpoint (final_step_*.pt or step_*.pt)
  --checkpoint-model ema            # 'ema' (default, smoother) or 'raw' weights
  --dataset <zarr>                  # source of start/goal pairs (and env id if not overridden)
  --source dataset|fresh            # 'dataset' replays stored pairs; 'fresh' samples new seeds
  --episodes 10                     # number of rollout episodes
  --episode-indices 1 5 9           # (dataset source) specific 1-indexed episodes instead of first N
  --env-id-override <env id>        # force env (needed for xArm7 fresh rollouts)
  --seed-start 20000                # first seed for fresh rollouts
  --max-steps 150                   # max control steps per episode
  --replan-stride N                 # execute N actions of each predicted chunk before replanning
  --post-success-steps 8            # extra steps recorded after success (matches hold-style stop)
  --action-ema-alpha 1.0            # 1.0 = no action smoothing; <1 smooths (0.1 = heavy)
  --video-fps 10                    # mp4 frame rate
  --device cuda                     # run policy + sim on GPU
  --output-dir <dir>                # where episode_*.mp4, episode_*.rrd, summary.json, metrics.jsonl land
```

## View the Rerun timeline

```bash
uv run rerun artifacts/reach-datasets/xarm7-pnp-cover-rollouts-fresh/episode_000.rrd
```

Use the `step` timeline in the Rerun viewer and press play.

## Videos of a dataset (re-simulate stored actions -> mp4 + rerun)

To render videos/Rerun of the *dataset itself* (not a policy), replay the stored
simulator actions:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/reach-datasets/pg3d-xarm7-pnp-cover.zarr \
  --episodes 6 \
  --video-dir artifacts/reach-datasets/pnp-cover-videos \
  --video-fps 15 \
  --rerun-dir artifacts/reach-datasets/pnp-cover-rerun

uv run rerun artifacts/reach-datasets/pnp-cover-rerun/episode_000.rrd
```

`--video-dir` and `--rerun-dir` are independent opt-ins (use either or both);
`--episode-indices 1 25 40` (1-indexed) picks specific episodes.

## Inspect the triad in the saved point cloud

The baked triad lives in the point cloud's trailing goal slots (not the rendered
sim scene), so it shows up in the Rerun `.rrd` and in the Plotly analyzer notebook
(`notebooks/zarr_pointcloud_analyzer.ipynb`), not in the mp4.
