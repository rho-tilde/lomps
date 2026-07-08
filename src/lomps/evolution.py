#!/usr/bin/env python
"""Resumable LOMPS evolution with fixed-target left-canonical LM restarts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np

from .canonical import (
    canonical_errors,
    polar_retraction,
    random_left_canonical,
    stack_tensor,
    unstack_tensor,
)
from .optimizer import LMOptions, optimize_tensor
from .protocol import NONINTEGRABLE_ISING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-A", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--base-time", type=float, required=True)
    parser.add_argument("--block-length", type=int, default=NONINTEGRABLE_ISING.block_length)
    parser.add_argument(
        "--odd-parity-warning-threshold",
        type=float,
        default=NONINTEGRABLE_ISING.odd_parity_warning_threshold,
    )
    parser.add_argument("--accept-cost", type=float, default=3e-16)
    parser.add_argument("--rank-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--fixed-point-solver",
        choices=("dense", "fast"),
        default="dense",
        help=(
            "Fixed-point solver used inside optimizer evaluations. "
            "'dense' is reproducible and default; 'fast' uses ARPACK first."
        ),
    )
    parser.add_argument("--perturb-amplitudes", type=str, default="0.3,0.6,1,2,4,8")
    parser.add_argument("--perturbations-per-amplitude", type=int, default=2)
    parser.add_argument("--random-restarts", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260702)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--run-time-limit", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, value)
    temporary.replace(path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def load_initial(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim == 4 and len(value) == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 2 or value.shape[1] != value.shape[2]:
        raise ValueError("--initial-A must contain A with shape (2,D,D), optionally batched once")
    return np.asarray(value, dtype=np.complex128)


def options(
    accept_cost: float,
    rank_tolerance: float,
    fixed_point_solver: str = "dense",
) -> tuple[LMOptions, LMOptions]:
    primary = LMOptions(
        max_iterations=40_000,
        gradient_tolerance=1e-11,
        cost_tolerance=accept_cost,
        rank_tolerance=rank_tolerance,
        fixed_point_solver=fixed_point_solver,
        plateau_window=50,
        plateau_relative_cost_drop=1e-4,
        plateau_absolute_cost_drop=1e-20,
        maximum_seconds=600.0,
        verbose=False,
    )
    strict = replace(
        primary,
        gradient_tolerance=1e-13,
        plateau_window=200,
        plateau_relative_cost_drop=1e-9,
        plateau_absolute_cost_drop=1e-24,
    )
    return primary, strict


def protocol_from_args(args: argparse.Namespace):
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    if args.odd_parity_warning_threshold < 0:
        raise ValueError("--odd-parity-warning-threshold must be non-negative")
    protocol = replace(
        NONINTEGRABLE_ISING,
        name=f"nonintegrable_ising_L{args.block_length}",
        block_length=args.block_length,
        odd_parity_warning_threshold=args.odd_parity_warning_threshold,
    )
    # Touch the derived property early so invalid protocols fail before any
    # checkpoint files are created.
    _ = protocol.lightcone_sites
    return protocol


def infer_block_length(seed: np.ndarray, target: np.ndarray) -> int:
    local_dimension = int(seed.shape[0])
    target_dimension = int(target.shape[0])
    block_length = 0
    dimension = 1
    while dimension < target_dimension:
        dimension *= local_dimension
        block_length += 1
    if dimension != target_dimension or target.shape[1] != target_dimension:
        raise ValueError("target shape is incompatible with the physical dimension")
    return block_length


def fit_fixed_target(
    seed: np.ndarray,
    target: np.ndarray,
    primary: LMOptions,
    strict: LMOptions,
) -> tuple[np.ndarray, object, bool, float]:
    """Fit one seed to an unchanged target, retrying strictly if needed."""

    started = time.perf_counter()
    block_length = infer_block_length(seed, target)
    best_A, best = optimize_tensor(seed, target, block_length, primary)
    used_strict = False
    if best.cost > primary.cost_tolerance:
        strict_A, strict_result = optimize_tensor(seed, target, block_length, strict)
        if strict_result.cost < best.cost:
            best_A, best = strict_A, strict_result
            used_strict = True
    return best_A, best, used_strict, time.perf_counter() - started


def distant_seeds(
    A: np.ndarray,
    *,
    step: int,
    amplitudes: tuple[float, ...],
    per_amplitude: int,
    random_restarts: int,
    random_seed: int,
) -> Iterator[tuple[str, float, int, np.ndarray, float]]:
    """Yield reproducible, left-canonical seeds far from the warm start."""

    d, D, _ = A.shape
    W = stack_tensor(A)
    rng = np.random.default_rng(random_seed + 1_000_003 * step)
    trial = 0
    for amplitude in amplitudes:
        for _ in range(per_amplitude):
            Z = (
                rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)
            ) / np.sqrt(2.0 * D)
            seed_W = polar_retraction(W + amplitude * Z)
            seed = unstack_tensor(seed_W, d, D)
            yield (
                "perturbed_warm_start",
                float(amplitude),
                trial,
                seed,
                float(np.linalg.norm(seed_W - W)),
            )
            trial += 1
    for index in range(random_restarts):
        seed, seed_W = random_left_canonical(
            d,
            D,
            seed=random_seed + 10_000_019 * step + index,
        )
        yield (
            "haar_random",
            float("nan"),
            trial,
            seed,
            float(np.linalg.norm(seed_W - W)),
        )
        trial += 1


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError("steps and checkpoint cadence must be positive")
    if args.accept_cost <= 0:
        raise ValueError("--accept-cost must be positive")
    protocol = protocol_from_args(args)
    amplitudes = tuple(float(value) for value in args.perturb_amplitudes.split(","))
    if not amplitudes or args.perturbations_per_amplitude < 0 or args.random_restarts < 0:
        raise ValueError("invalid restart counts or amplitudes")
    initial = load_initial(args.initial_A)
    primary, strict = options(
        args.accept_cost,
        args.rank_tolerance,
        args.fixed_point_solver,
    )
    policy = {
        "protocol": asdict(protocol),
        "lightcone_sites": protocol.lightcone_sites,
        "target_margins": protocol.target_margins,
        "fixed_target": True,
        "fixed_point_solver": args.fixed_point_solver,
        "accept_cost": args.accept_cost,
        "perturb_amplitudes": list(amplitudes),
        "perturbations_per_amplitude": args.perturbations_per_amplitude,
        "random_restarts": args.random_restarts,
        "random_seed": args.random_seed,
        "primary_optimizer": asdict(primary),
        "strict_optimizer": asdict(strict),
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "states.npy"
    times_path = output / "times.npy"
    steps_path = output / "steps.csv"
    restarts_path = output / "restart_trials.csv"
    metadata_path = output / "metadata.json"
    pause_path = output / "PAUSE"

    if args.resume:
        metadata = json.loads(metadata_path.read_text())
        if metadata["policy"] != policy or metadata["steps"] != args.steps:
            raise ValueError("run policy or requested length differs from checkpoint")
        completed = int(metadata["completed_steps"])
        states = np.lib.format.open_memmap(states_path, mode="r+")
        A = states[completed].copy()
        metadata["status"] = "running"
        metadata["resumed_at"] = utc_now()
        metadata.pop("failure", None)
    else:
        if states_path.exists() or metadata_path.exists():
            raise FileExistsError("output exists; use --resume or a new output directory")
        states = np.lib.format.open_memmap(
            states_path,
            mode="w+",
            dtype=np.complex128,
            shape=(args.steps + 1, *initial.shape),
        )
        states[0] = initial
        states.flush()
        atomic_npy(
            times_path,
            args.base_time + protocol.delta_t * np.arange(args.steps + 1),
        )
        completed = 0
        A = initial.copy()
        metadata = {
            "status": "running",
            "created_at": utc_now(),
            "initial_A": str(args.initial_A.resolve()),
            "steps": args.steps,
            "base_time": args.base_time,
            "delta_t": protocol.delta_t,
            "final_time": args.base_time + args.steps * protocol.delta_t,
            "completed_steps": 0,
            "policy": policy,
        }
        atomic_json(metadata_path, metadata)

    invocation_started = time.perf_counter()
    cumulative_seconds = float(metadata.get("cumulative_seconds", 0.0))
    rescue_events = int(metadata.get("rescue_events", 0))
    status = "completed"
    try:
        for step in range(completed + 1, args.steps + 1):
            if pause_path.exists():
                status = "paused_by_PAUSE_file"
                break
            if args.run_time_limit > 0 and time.perf_counter() - invocation_started >= args.run_time_limit:
                status = "paused_by_run_time_limit"
                break
            started = time.perf_counter()
            # This target is computed exactly once.  Every rescue trial below
            # minimizes against this unchanged matrix.
            target = protocol.target_rdm(A)
            next_A, result, used_strict, fit_seconds = fit_fixed_target(
                A, target, primary, strict
            )
            warm_cost = float(result.cost)
            restart_used = False
            accepted_kind = "warm_start"
            accepted_trial = -1
            trials = 0

            if result.cost > args.accept_cost:
                for kind, amplitude, trial, seed, seed_distance in distant_seeds(
                    A,
                    step=step,
                    amplitudes=amplitudes,
                    per_amplitude=args.perturbations_per_amplitude,
                    random_restarts=args.random_restarts,
                    random_seed=args.random_seed,
                ):
                    candidate_A, candidate_result, candidate_strict, seconds = fit_fixed_target(
                        seed, target, primary, strict
                    )
                    trials += 1
                    append_csv(
                        restarts_path,
                        {
                            "step": step,
                            "time": args.base_time + step * protocol.delta_t,
                            "target_frozen": True,
                            "trial": trial,
                            "seed_kind": kind,
                            "amplitude": amplitude,
                            "seed_stiefel_distance": seed_distance,
                            "seed_left_canonical_error": canonical_errors(seed)[
                                "left_canonical_error"
                            ],
                            "cost": candidate_result.cost,
                            "target_residual": candidate_result.residual_norm,
                            "status": candidate_result.status,
                            "evaluations": len(candidate_result.history),
                            "used_strict_retry": candidate_strict,
                            "seconds": seconds,
                            "accepted": candidate_result.cost <= args.accept_cost,
                        },
                    )
                    print(
                        f"restart step={step} trial={trial} kind={kind} "
                        f"amplitude={amplitude:g} distance={seed_distance:.3f} "
                        f"cost={candidate_result.cost:.2e}",
                        flush=True,
                    )
                    if candidate_result.cost <= args.accept_cost:
                        next_A, result = candidate_A, candidate_result
                        restart_used = True
                        accepted_kind = kind
                        accepted_trial = trial
                        rescue_events += 1
                        break
                    if candidate_result.cost < result.cost:
                        next_A, result = candidate_A, candidate_result

            step_seconds = time.perf_counter() - started
            if result.cost > args.accept_cost:
                status = "failed_fixed_target_multistart"
                metadata["failure"] = {
                    "attempted_step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "warm_start_cost": warm_cost,
                    "best_cost": result.cost,
                    "best_target_residual": result.residual_norm,
                    "restart_trials": trials,
                }
                break

            A = next_A
            states[step] = A
            completed = step
            cumulative_seconds += step_seconds
            append_csv(
                steps_path,
                {
                    "step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "warm_start_cost": warm_cost,
                    "optimizer_cost": result.cost,
                    "target_residual": result.residual_norm,
                    "optimizer_status": result.status,
                    "optimizer_evaluations": len(result.history),
                    "warm_used_strict_retry": used_strict,
                    "restart_used": restart_used,
                    "accepted_seed_kind": accepted_kind,
                    "accepted_restart_trial": accepted_trial,
                    "restart_trials": trials,
                    "fit_seconds_before_restarts": fit_seconds,
                    "step_seconds": step_seconds,
                    "left_canonical_error": canonical_errors(A)["left_canonical_error"],
                },
            )
            if step % args.checkpoint_every == 0 or restart_used or step == args.steps:
                states.flush()
                metadata.update(
                    {
                        "status": "running",
                        "updated_at": utc_now(),
                        "completed_steps": completed,
                        "rescue_events": rescue_events,
                        "cumulative_seconds": cumulative_seconds,
                    }
                )
                atomic_json(metadata_path, metadata)
            if step <= 3 or step % 10 == 0 or restart_used:
                print(
                    f"step={step}/{args.steps} time={args.base_time + step * protocol.delta_t:.3f} "
                    f"cost={result.cost:.2e} restart={restart_used} trials={trials} "
                    f"wall={step_seconds:.2f}s",
                    flush=True,
                )
    except KeyboardInterrupt:
        status = "paused_by_keyboard_interrupt"
    finally:
        states.flush()
        if completed == args.steps:
            status = "completed"
        metadata.update(
            {
                "status": status,
                "updated_at": utc_now(),
                "completed_steps": completed,
                "rescue_events": rescue_events,
                "cumulative_seconds": cumulative_seconds,
                "run_seconds_latest_invocation": time.perf_counter() - invocation_started,
            }
        )
        atomic_json(metadata_path, metadata)
        print(f"run_status={status} completed={completed}/{args.steps}", flush=True)


if __name__ == "__main__":
    main()
