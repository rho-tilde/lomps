"""Run a LOMPS trajectory with monotone bond-dimension promotion.

Each bond dimension is handled by an ordinary ``lomps-run`` child process.
When a child exhausts its optimizer without accepting the next update, the
last accepted tensor becomes the exact source for a new, higher-D segment.
The child runner therefore constructs the next physical target from the old
tensor while using its noisy lift only as an optimizer seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np


PROMOTABLE_STATUSES = {"failed_fixed_target_multistart"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_bond_dimensions(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("at least one bond dimension is required")
    if any(value < 1 for value in values):
        raise ValueError("bond dimensions must be positive")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("bond dimensions must be strictly increasing")
    return values


def extract_accepted_checkpoint(
    segment_dir: Path,
    completed_steps: int,
    tensor_path: Path,
    fixed_point_path: Path,
) -> None:
    states = np.load(segment_dir / "states.npy", mmap_mode="r")
    fixed_points = np.load(segment_dir / "right_fixed_points.npy", mmap_mode="r")
    index = int(completed_steps)
    if not 0 <= index < states.shape[0]:
        raise IndexError("completed step lies outside the saved state array")
    np.save(tensor_path, np.asarray(states[index]))
    np.save(fixed_point_path, np.asarray(fixed_points[index]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LOMPS at the smallest requested D and promote to the next D "
            "only after the current segment cannot accept its next update."
        )
    )
    parser.add_argument("--initial-A", type=Path, required=True)
    parser.add_argument("--initial-key", default=None)
    parser.add_argument(
        "--initial-layout",
        choices=("auto", "physical-left-right", "legacy-left-physical-right"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bond-dimensions", required=True)
    parser.add_argument("--block-length", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--base-time", type=float, default=0.0)
    parser.add_argument("--delta-t", type=float, default=1e-3)
    parser.add_argument("--protocol", default="nonintegrable-ising")
    parser.add_argument("--g", type=float, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--J", type=float, default=None)
    parser.add_argument("--trotter-order", type=int, default=None)
    parser.add_argument(
        "--symmetric-transverse",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--accept-cost", type=float, default=1e-14)
    parser.add_argument("--initial-first-step-accept-cost", type=float, default=1e-15)
    parser.add_argument("--handoff-accept-cost", type=float, default=1e-14)
    parser.add_argument("--embedding-noise-amplitude", type=float, default=1e-6)
    parser.add_argument("--embedding-seed", type=int, default=104_729)
    parser.add_argument(
        "--first-step-optimizer",
        choices=("cg-lm", "lm"),
        default="cg-lm",
        help="Optimizer used to fit each newly lifted bond-dimension handoff.",
    )
    parser.add_argument("--first-step-cg-seconds", type=float, default=900.0)
    parser.add_argument("--fixed-point-solver", choices=("dense", "fast"), default="dense")
    parser.add_argument(
        "--target-source-fixed-point-solver",
        choices=("dense", "fast"),
        default="dense",
    )
    parser.add_argument("--target-contraction", choices=("tensor", "dense"), default="tensor")
    parser.add_argument("--perturb-amplitudes", default="0.1,0.3,0.6,1.0")
    parser.add_argument("--perturbations-per-amplitude", type=int, default=1)
    parser.add_argument("--random-restarts", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260702)
    parser.add_argument(
        "--strict-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser


def child_command(
    args: argparse.Namespace,
    *,
    source: Path,
    source_layout: str,
    segment_dir: Path,
    bond_dimension: int,
    base_time: float,
    steps: int,
    first_step_accept_cost: float,
    include_initial_key: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lomps.evolution",
        "--protocol",
        args.protocol,
        "--initial-A",
        str(source),
        "--initial-layout",
        source_layout,
        "--output-dir",
        str(segment_dir),
        "--bond-dimension",
        str(bond_dimension),
        "--block-length",
        str(args.block_length),
        "--steps",
        str(steps),
        "--base-time",
        repr(base_time),
        "--delta-t",
        repr(args.delta_t),
        "--accept-cost",
        repr(args.accept_cost),
        "--first-step-accept-cost",
        repr(first_step_accept_cost),
        "--embedding-noise-amplitude",
        repr(args.embedding_noise_amplitude),
        "--embedding-seed",
        str(args.embedding_seed),
        "--first-step-optimizer",
        args.first_step_optimizer,
        "--first-step-cg-seconds",
        repr(args.first_step_cg_seconds),
        "--fixed-point-solver",
        args.fixed_point_solver,
        "--target-source-fixed-point-solver",
        args.target_source_fixed_point_solver,
        "--target-contraction",
        args.target_contraction,
        "--perturb-amplitudes",
        args.perturb_amplitudes,
        "--perturbations-per-amplitude",
        str(args.perturbations_per_amplitude),
        "--random-restarts",
        str(args.random_restarts),
        "--random-seed",
        str(args.random_seed),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--strict-retry" if args.strict_retry else "--no-strict-retry",
    ]
    if include_initial_key and args.initial_key is not None:
        command.extend(("--initial-key", args.initial_key))
    for option, value in (
        ("--g", args.g),
        ("--h", args.h),
        ("--J", args.J),
        ("--trotter-order", args.trotter_order),
    ):
        if value is not None:
            command.extend((option, str(value)))
    if args.symmetric_transverse is not None:
        command.append(
            "--symmetric-transverse"
            if args.symmetric_transverse
            else "--no-symmetric-transverse"
        )
    return command


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ladder = parse_bond_dimensions(args.bond_dimensions)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    if args.delta_t <= 0:
        raise ValueError("--delta-t must be positive")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("adaptive output exists; choose a new directory")
    output.mkdir(parents=True)
    handoff_dir = output / "handoffs"
    handoff_dir.mkdir()
    manifest_path = output / "manifest.json"

    source = args.initial_A.resolve()
    source_layout = args.initial_layout
    accepted_steps = 0
    base_time = float(args.base_time)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "initial_A": str(source),
        "bond_dimensions": list(ladder),
        "block_length": args.block_length,
        "delta_t": args.delta_t,
        "requested_steps": args.steps,
        "accepted_steps": 0,
        "current_time": base_time,
        "segments": [],
    }
    atomic_json(manifest_path, manifest)

    for segment_index, bond_dimension in enumerate(ladder):
        remaining = args.steps - accepted_steps
        if remaining <= 0:
            break
        segment_dir = output / f"segment_{segment_index:03d}_D{bond_dimension}"
        threshold = (
            args.initial_first_step_accept_cost
            if segment_index == 0
            else args.handoff_accept_cost
        )
        command = child_command(
            args,
            source=source,
            source_layout=source_layout,
            segment_dir=segment_dir,
            bond_dimension=bond_dimension,
            base_time=base_time,
            steps=remaining,
            first_step_accept_cost=threshold,
            include_initial_key=segment_index == 0,
        )
        print(
            f"adaptive segment={segment_index} D={bond_dimension} "
            f"base_time={base_time:.12g} remaining_steps={remaining}",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            manifest["status"] = "child_process_error"
            manifest["returncode"] = completed.returncode
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            raise SystemExit(completed.returncode)

        metadata = json.loads((segment_dir / "metadata.json").read_text())
        segment_steps = int(metadata["completed_steps"])
        status = str(metadata["status"])
        segment_record = {
            "index": segment_index,
            "bond_dimension": bond_dimension,
            "directory": segment_dir.name,
            "base_time": base_time,
            "requested_steps": remaining,
            "completed_steps": segment_steps,
            "status": status,
            "failure": metadata.get("failure"),
        }
        manifest["segments"].append(segment_record)

        if segment_steps > 0:
            tensor_path = handoff_dir / f"handoff_{segment_index:03d}_D{bond_dimension}.npy"
            fixed_point_path = (
                handoff_dir / f"handoff_{segment_index:03d}_D{bond_dimension}_rfp.npy"
            )
            extract_accepted_checkpoint(
                segment_dir,
                segment_steps,
                tensor_path,
                fixed_point_path,
            )
            source = tensor_path
            source_layout = "physical-left-right"
            accepted_steps += segment_steps
            base_time = args.base_time + accepted_steps * args.delta_t
            segment_record["handoff_tensor"] = str(tensor_path.relative_to(output))
            segment_record["handoff_right_fixed_point"] = str(
                fixed_point_path.relative_to(output)
            )

        manifest["accepted_steps"] = accepted_steps
        manifest["current_time"] = base_time
        manifest["updated_at"] = utc_now()

        if accepted_steps == args.steps:
            manifest["status"] = "completed"
            atomic_json(manifest_path, manifest)
            print(
                f"adaptive_status=completed accepted={accepted_steps}/{args.steps} "
                f"time={base_time:.12g} D={bond_dimension}",
                flush=True,
            )
            return
        if status not in PROMOTABLE_STATUSES:
            manifest["status"] = status
            atomic_json(manifest_path, manifest)
            print(
                f"adaptive_status={status} accepted={accepted_steps}/{args.steps}",
                flush=True,
            )
            return
        atomic_json(manifest_path, manifest)

    manifest["status"] = "exhausted_bond_dimensions"
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    print(
        f"adaptive_status=exhausted_bond_dimensions "
        f"accepted={accepted_steps}/{args.steps} time={base_time:.12g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
