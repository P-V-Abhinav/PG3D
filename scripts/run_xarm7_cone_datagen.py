#!/usr/bin/env python3
"""Single-script data generation for the XArm7 full-cone orientation dataset.

Generates 300 demos for each of 25 orientation modes (1 downward + 3 tilt levels
× 8 azimuths = 25 modes), writing them sequentially into a single Zarr dataset.
All modes use EMA smoothing (alpha=0.6) so the training distribution matches
inference.

Usage on the server:
    python scripts/run_xarm7_cone_datagen.py \
        --output /scratch2/abhinav.pv/PG3D_artifacts/xarm7_cone.zarr \
        --num-workers 16
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ── cone mode list (must match _cone_orientation_names in write_maniskill_pose_reach_dataset.py) ──
import numpy as np

def _cone_mode_names(max_tilt_deg: float = 60.0, tilt_steps: int = 3, azimuth_steps: int = 8) -> list[str]:
    names = ["downward"]
    tilt_angles = np.linspace(0, max_tilt_deg, tilt_steps + 1)[1:]
    azimuth_angles = np.linspace(0, 360, azimuth_steps, endpoint=False)
    for theta_deg in tilt_angles:
        for phi_deg in azimuth_angles:
            names.append(f"cone_t{int(round(theta_deg)):02d}_p{int(round(phi_deg)):03d}")
    return names


ALL_MODES = _cone_mode_names()  # 25 modes total


def _run_mode(
    mode: str,
    output: Path,
    num_demos: int,
    num_workers: int,
    seed_start: int,
    is_first: bool,
    extra_args: list[str],
) -> int:
    """Run one orientation mode. Returns the subprocess exit code."""
    write_flag = "--overwrite" if is_first else "--append"

    cmd = [
        sys.executable,
        "scripts/write_xarm7_pose_reach_dataset.py",
        "--orientation-mode", mode,
        "--num-demos", str(num_demos),
        "--num-workers", str(num_workers),
        "--output", str(output),
        "--action-ema-alpha", "0.6",
        "--seed-start", str(seed_start),
        write_flag,
        *extra_args,
    ]

    print(f"\n{'='*70}")
    print(f"[cone datagen]  mode={mode!r}  target={num_demos} demos  {'CREATE' if is_first else 'APPEND'}")
    print(f"{'='*70}")
    print("CMD:", " ".join(cmd), flush=True)

    result = subprocess.run(cmd, check=False)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a balanced XArm7 cone-orientation dataset in one shot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/scratch2/abhinav.pv/PG3D_artifacts/xarm7_cone.zarr"),
        help="Path to the output Zarr file.",
    )
    parser.add_argument(
        "--num-demos", type=int, default=300,
        help="Number of successful demos to collect per orientation mode (default: 300).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=16,
        help="Number of Ray parallel workers.",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="Starting seed. Each mode advances the seed by --num-demos to avoid overlap.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=None,
        help=(
            "Subset of modes to run (e.g. --modes downward cone_t20_p000). "
            "Default: all 25 cone modes in order."
        ),
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help=(
            "Resume generation starting from this mode name (inclusive). "
            "The first resumed mode will use --append, not --overwrite."
        ),
    )
    # Passthrough args for write_xarm7_pose_reach_dataset.py
    parser.add_argument(
        "--extra", nargs=argparse.REMAINDER, default=[],
        help="Any extra flags to pass verbatim to write_xarm7_pose_reach_dataset.py.",
    )
    args = parser.parse_args(argv)

    modes_to_run = args.modes if args.modes is not None else ALL_MODES

    # Handle resume: find the starting index in ALL_MODES
    start_idx = 0
    resume_is_first = False  # when resuming, the zarr already exists → use --append
    if args.resume_from is not None:
        if args.resume_from not in modes_to_run:
            print(f"ERROR: --resume-from mode {args.resume_from!r} not in mode list.", file=sys.stderr)
            return 1
        start_idx = modes_to_run.index(args.resume_from)
        resume_is_first = False
        print(f"[cone datagen] Resuming from mode index {start_idx}: {args.resume_from!r}")

    modes_to_run = modes_to_run[start_idx:]
    total = len(modes_to_run)
    failures: list[str] = []

    for i, mode in enumerate(modes_to_run):
        # is_first: create the zarr for the very first mode (unless resuming into an existing file)
        is_first = (i == 0) and (args.resume_from is None)
        # Each mode gets a distinct seed range so goals don't repeat between modes.
        seed_start_for_mode = args.seed_start + i * args.num_demos

        rc = _run_mode(
            mode=mode,
            output=args.output,
            num_demos=args.num_demos,
            num_workers=args.num_workers,
            seed_start=seed_start_for_mode,
            is_first=is_first,
            extra_args=args.extra,
        )

        print(f"\n[cone datagen] Finished mode {i+1}/{total}: {mode!r}  exit_code={rc}", flush=True)
        if rc != 0:
            failures.append(mode)
            print(
                f"[cone datagen] WARNING: mode {mode!r} exited with code {rc}. "
                "Continuing to next mode.",
                flush=True,
            )

    print(f"\n{'='*70}")
    print(f"[cone datagen] ALL DONE")
    print(f"  Total modes attempted : {total}")
    print(f"  Failures              : {len(failures)}")
    if failures:
        print(f"  Failed modes: {failures}")
    print(f"  Output zarr           : {args.output}")
    print(f"{'='*70}\n")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
