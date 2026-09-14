#!/usr/bin/env python3
"""Run a resumable consecutive L=4 trajectory with converged P8 selection.

The saved L=4,D=12 trajectory supplies only the starting anchor and optional
diagnostics.  The anchor is first minimized in its fixed-rho4 fibre.  Every
subsequent target is then constructed from the previously refined state, fit
to the unchanged exact rho4 threshold, and refined until a configured purity
convergence criterion is met.  No continuity penalty is used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from lomps.buffered_purity import (
    BufferedPurityOptions,
    BufferedPurityResult,
    density_matrix_purity,
    minimize_buffered_purity,
)
from lomps.canonical import canonical_errors
from lomps.optimizer import LMOptions, optimize_tensor, optimizer_right_fixed_point
from lomps.predictor import trajectory_secant_predictor
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm


Array = np.ndarray
CONVERGED_PURITY_STATUSES = {
    "null_gradient_tolerance",
    "relative_purity_tolerance",
}
CSV_FIELDS = [
    "local_step",
    "time",
    "ordinary_fit_cost",
    "ordinary_fit_status",
    "ordinary_fit_evaluations",
    "purity_status",
    "purity_iterations",
    "purity_before",
    "purity_after",
    "relative_purity_drop",
    "terminal_null_gradient_norm",
    "terminal_relative_purity_drop",
    "branch_to_baseline_rho4_trace_distance",
    "branch_to_l5_rho4_trace_distance",
    "baseline_to_l5_rho4_trace_distance",
    "fit_seconds",
    "purity_seconds",
    "step_seconds",
    "left_canonical_error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_checkpoint(
    path: Path,
    *,
    previous: Array,
    current: Array,
    completed_steps: int,
    current_time: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            previous=np.asarray(previous, dtype=np.complex128),
            current=np.asarray(current, dtype=np.complex128),
            completed_steps=np.asarray(completed_steps, dtype=np.int64),
            current_time=np.asarray(current_time, dtype=float),
        )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--start-time", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--accept-cost", type=float, default=3e-16)
    parser.add_argument("--purity-step", type=float, default=50.0)
    parser.add_argument("--purity-max-iterations", type=int, default=2000)
    parser.add_argument(
        "--purity-gradient-tolerance", type=float, default=1e-8
    )
    parser.add_argument(
        "--purity-relative-tolerance", type=float, default=1e-12
    )
    parser.add_argument("--jacobian-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-purity", action="store_true")
    return parser.parse_args()


def trace_distance(rho: Array, sigma: Array) -> float:
    difference = 0.5 * ((rho - sigma) + (rho - sigma).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


class References:
    def __init__(self, data_dir: Path):
        l4_root = data_dir / "nonintegrable_d12_trajectory"
        self.l4_initial = np.asarray(
            np.load(data_dir / "nonintegrable_d12_t0p001.npy"),
            dtype=np.complex128,
        )
        self.l4_states = np.load(
            l4_root / "trajectory_states.npy", mmap_mode="r"
        )
        l5_root = (
            data_dir
            / "trajectories"
            / "l5_d20d21_stitched_t0p001_to_t7p189"
        )
        self.l5_times_d20 = np.load(l5_root / "times_D20.npy", mmap_mode="r")
        self.l5_times_d21 = np.load(l5_root / "times_D21.npy", mmap_mode="r")
        self.l5_states_d20 = np.load(
            l5_root / "states_D20.npy", mmap_mode="r"
        )
        self.l5_states_d21 = np.load(
            l5_root / "states_D21.npy", mmap_mode="r"
        )

    def l4(self, physical_step: int) -> Array:
        if physical_step == 1:
            return self.l4_initial.copy()
        if physical_step < 1 or physical_step > len(self.l4_states) + 1:
            raise ValueError(f"no L=4 reference at step {physical_step}")
        return np.asarray(self.l4_states[physical_step - 2], dtype=np.complex128)

    def l5(self, time_value: float) -> Array | None:
        if time_value <= float(self.l5_times_d20[-1]):
            times, states = self.l5_times_d20, self.l5_states_d20
        else:
            times, states = self.l5_times_d21, self.l5_states_d21
        index = int(round((time_value - float(times[0])) / 1e-3))
        if index < 0 or index >= len(times):
            return None
        if abs(float(times[index]) - time_value) > 1e-12:
            return None
        return np.asarray(states[index], dtype=np.complex128)


def exact_rdms(A: Array) -> tuple[Array, Array]:
    r, _ = optimizer_right_fixed_point(A, "dense")
    return block_rdm(A, 4, r), block_rdm(A, 8, r)


def dense_primary_cost(A: Array, target: Array) -> float:
    rho4, _ = exact_rdms(A)
    residual = rho4 - target
    return 0.5 * float(np.vdot(residual, residual).real)


def purity_options(args: argparse.Namespace) -> BufferedPurityOptions:
    return BufferedPurityOptions(
        buffer_sites=4,
        primary_cost_tolerance=args.accept_cost,
        max_iterations=args.purity_max_iterations,
        initial_step=args.purity_step,
        direction_scaling="gradient",
        minimum_step=1e-10,
        null_gradient_tolerance=args.purity_gradient_tolerance,
        relative_purity_tolerance=args.purity_relative_tolerance,
        reproject_primary=True,
        projection_max_iterations=8,
        projection_initial_damping=1e-10,
        projection_maximum_seconds=120.0,
        tangent_slice="grassmann",
        fixed_point_solver="dense",
        jacobian_response_solver="dense_lu",
        jacobian_workers=args.jacobian_workers,
        verbose=not args.quiet_purity,
    )


def primary_options(args: argparse.Namespace) -> LMOptions:
    return LMOptions(
        max_iterations=40,
        gradient_tolerance=1e-11,
        cost_tolerance=args.accept_cost,
        rank_tolerance=1e-10,
        tangent_slice="grassmann",
        initial_damping=1e-10,
        maximum_seconds=120.0,
        fixed_point_solver="dense",
        linear_solver="normal",
        rdm_vectorization="hermitian",
        jacobian_response_solver="dense_lu",
        jacobian_workers=args.jacobian_workers,
        verbose=False,
    )


def require_converged(result: BufferedPurityResult, label: str) -> None:
    if result.status not in CONVERGED_PURITY_STATUSES:
        raise RuntimeError(
            f"{label} purity refinement did not converge: "
            f"status={result.status}, accepted={result.accepted_steps}, "
            f"|g_null|={result.final_null_gradient_norm:.6e}"
        )


def append_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def reconcile_steps_csv(path: Path, completed_steps: int) -> None:
    """Discard only rows written after the durable checkpoint."""

    if not path.exists():
        if completed_steps:
            raise RuntimeError("checkpoint has steps but steps.csv is missing")
        return
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    durable = [
        row for row in rows if int(row["local_step"]) <= completed_steps
    ]
    expected = list(range(1, completed_steps + 1))
    observed = [int(row["local_step"]) for row in durable]
    if observed != expected:
        raise RuntimeError(
            f"steps.csv/checkpoint mismatch: expected {expected}, got {observed}"
        )
    if len(durable) == len(rows):
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(durable)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def update_manifest(
    path: Path, manifest: dict[str, Any], **updates: Any
) -> None:
    manifest.update(updates)
    manifest["updated_at"] = utc_now()
    atomic_json(path, manifest)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.purity_max_iterations < 1:
        raise ValueError("--purity-max-iterations must be positive")

    protocol = replace(
        NONINTEGRABLE_ISING,
        block_length=4,
        target_source_fixed_point_solver="dense",
    )
    start_step = int(round(args.start_time / protocol.delta_t))
    if abs(start_step * protocol.delta_t - args.start_time) > 1e-12:
        raise ValueError("--start-time must lie on the dt grid")

    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    checkpoint_path = output / "checkpoint.npz"
    steps_path = output / "steps.csv"
    states_dir = output / "states"
    p_options = purity_options(args)
    f_options = primary_options(args)
    config = {
        "protocol": "nonintegrable_ising_L4",
        "block_length": 4,
        "buffer_sites": 4,
        "extended_block_length": 8,
        "bond_dimension": 12,
        "delta_t": protocol.delta_t,
        "start_time": args.start_time,
        "target_steps": args.steps,
        "accept_cost": args.accept_cost,
        "primary_options": asdict(f_options),
        "purity_options": asdict(p_options),
        "no_continuity_penalty": True,
    }
    references = References(args.data_dir)

    if args.resume:
        if not manifest_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError("resume requires manifest.json and checkpoint.npz")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config") != config:
            raise RuntimeError("stored run configuration does not match arguments")
        with np.load(checkpoint_path) as saved:
            previous = np.asarray(saved["previous"], dtype=np.complex128)
            current = np.asarray(saved["current"], dtype=np.complex128)
            completed_steps = int(saved["completed_steps"])
            current_time = float(saved["current_time"])
        if completed_steps > args.steps:
            raise RuntimeError("checkpoint is beyond requested target step")
        states_dir.mkdir(exist_ok=True)
        reconcile_steps_csv(steps_path, completed_steps)
        update_manifest(
            manifest_path,
            manifest,
            status="running",
            pid=os.getpid(),
            resumed_at=utc_now(),
            current_stage="resumed",
            error=None,
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
        states_dir.mkdir()
        previous = references.l4(start_step - 1)
        baseline_anchor = references.l4(start_step)
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "pid": os.getpid(),
            "config": config,
            "completed_steps": 0,
            "current_time": args.start_time,
            "current_stage": "anchor_purity_refinement",
        }
        atomic_json(manifest_path, manifest)

        try:
            anchor_target = protocol.target_rdm(previous)
            baseline_anchor_cost = dense_primary_cost(
                baseline_anchor, anchor_target
            )
            if baseline_anchor_cost > args.accept_cost:
                raise RuntimeError(
                    "saved anchor is not feasible: "
                    f"{baseline_anchor_cost:.6e} > {args.accept_cost:.6e}"
                )
            anchor_started = time.perf_counter()
            current, anchor_result = minimize_buffered_purity(
                baseline_anchor, anchor_target, 4, p_options
            )
            anchor_seconds = time.perf_counter() - anchor_started
            require_converged(anchor_result, "anchor")
            anchor_cost = dense_primary_cost(current, anchor_target)
            if anchor_cost > args.accept_cost:
                raise RuntimeError(
                    "converged anchor violates the primary constraint"
                )
            np.save(
                states_dir / f"step_{0:06d}_t{args.start_time:.3f}.npy",
                current,
            )
            atomic_checkpoint(
                checkpoint_path,
                previous=previous,
                current=current,
                completed_steps=0,
                current_time=args.start_time,
            )
            update_manifest(
                manifest_path,
                manifest,
                anchor={
                    "baseline_primary_cost": baseline_anchor_cost,
                    "initial_purity": anchor_result.initial_purity,
                    "final_primary_cost": anchor_cost,
                    "final_purity": anchor_result.purity,
                    "relative_purity_drop": (
                        anchor_result.initial_purity - anchor_result.purity
                    )
                    / anchor_result.initial_purity,
                    "status": anchor_result.status,
                    "accepted_iterations": anchor_result.accepted_steps,
                    "terminal_null_gradient_norm": (
                        anchor_result.final_null_gradient_norm
                    ),
                    "terminal_relative_purity_drop": (
                        anchor_result.final_relative_purity_drop
                    ),
                    "seconds": anchor_seconds,
                },
                current_stage="physical_step",
            )
        except BaseException as exc:
            update_manifest(
                manifest_path,
                manifest,
                status="failed",
                current_stage="failed",
                error=f"{type(exc).__name__}: {exc}",
                failed_at=utc_now(),
            )
            raise
        completed_steps = 0
        current_time = args.start_time

    try:
        for local_step in range(completed_steps + 1, args.steps + 1):
            step_started = time.perf_counter()
            next_time = args.start_time + local_step * protocol.delta_t
            update_manifest(
                manifest_path,
                manifest,
                status="running",
                current_stage="ordinary_fit",
                active_local_step=local_step,
                target_time=next_time,
            )
            target = protocol.target_rdm(current)
            # The anchor may have moved substantially inside its fibre, so do
            # not extrapolate the old saved trajectory on the first new step.
            seed = (
                current.copy()
                if local_step == 1
                else trajectory_secant_predictor(previous, current)
            )
            fit_started = time.perf_counter()
            fitted, fit_result = optimize_tensor(seed, target, 4, f_options)
            fit_seconds = time.perf_counter() - fit_started
            fitted_cost = dense_primary_cost(fitted, target)
            if fitted_cost > args.accept_cost:
                raise RuntimeError(
                    f"ordinary fit failed at local step {local_step}: "
                    f"{fitted_cost:.6e} > {args.accept_cost:.6e}"
                )
            _, fitted_rho8 = exact_rdms(fitted)
            purity_before = density_matrix_purity(fitted_rho8)

            update_manifest(
                manifest_path,
                manifest,
                current_stage="purity_refinement",
                ordinary_fit_cost=fitted_cost,
                ordinary_fit_status=fit_result.status,
                ordinary_fit_evaluations=len(fit_result.history),
            )
            purity_started = time.perf_counter()
            refined, purity_result = minimize_buffered_purity(
                fitted, target, 4, p_options
            )
            purity_seconds = time.perf_counter() - purity_started
            require_converged(purity_result, f"step {local_step}")
            refined_rho4, refined_rho8 = exact_rdms(refined)
            refined_cost = 0.5 * float(
                np.vdot(refined_rho4 - target, refined_rho4 - target).real
            )
            if refined_cost > args.accept_cost:
                raise RuntimeError(
                    f"refined step {local_step} violates primary cost: "
                    f"{refined_cost:.6e}"
                )
            purity_after = density_matrix_purity(refined_rho8)

            baseline = references.l4(start_step + local_step)
            baseline_rho4, _ = exact_rdms(baseline)
            l5 = references.l5(next_time)
            if l5 is None:
                branch_l5 = np.nan
                baseline_l5 = np.nan
            else:
                l5_rho4, _ = exact_rdms(l5)
                branch_l5 = trace_distance(refined_rho4, l5_rho4)
                baseline_l5 = trace_distance(baseline_rho4, l5_rho4)
            row = {
                "local_step": local_step,
                "time": next_time,
                "ordinary_fit_cost": fitted_cost,
                "ordinary_fit_status": fit_result.status,
                "ordinary_fit_evaluations": len(fit_result.history),
                "purity_status": purity_result.status,
                "purity_iterations": purity_result.accepted_steps,
                "purity_before": purity_before,
                "purity_after": purity_after,
                "relative_purity_drop": (
                    purity_before - purity_after
                )
                / purity_before,
                "terminal_null_gradient_norm": (
                    purity_result.final_null_gradient_norm
                ),
                "terminal_relative_purity_drop": (
                    purity_result.final_relative_purity_drop
                ),
                "branch_to_baseline_rho4_trace_distance": trace_distance(
                    refined_rho4, baseline_rho4
                ),
                "branch_to_l5_rho4_trace_distance": branch_l5,
                "baseline_to_l5_rho4_trace_distance": baseline_l5,
                "fit_seconds": fit_seconds,
                "purity_seconds": purity_seconds,
                "step_seconds": time.perf_counter() - step_started,
                "left_canonical_error": canonical_errors(refined)[
                    "left_canonical_error"
                ],
            }
            append_row(steps_path, row)
            np.save(
                states_dir / f"step_{local_step:06d}_t{next_time:.3f}.npy",
                refined,
            )
            old_current = current
            previous, current = old_current, refined
            completed_steps = local_step
            current_time = next_time
            atomic_checkpoint(
                checkpoint_path,
                previous=previous,
                current=current,
                completed_steps=completed_steps,
                current_time=current_time,
            )
            update_manifest(
                manifest_path,
                manifest,
                completed_steps=completed_steps,
                current_time=current_time,
                current_stage="checkpointed",
                latest_step=row,
            )
            print("STEP " + json.dumps(row, sort_keys=True), flush=True)

        update_manifest(
            manifest_path,
            manifest,
            status="completed",
            current_stage="completed",
            completed_steps=completed_steps,
            current_time=current_time,
            finished_at=utc_now(),
        )
    except BaseException as exc:
        update_manifest(
            manifest_path,
            manifest,
            status="failed",
            current_stage="failed",
            error=f"{type(exc).__name__}: {exc}",
            failed_at=utc_now(),
            completed_steps=completed_steps,
            current_time=current_time,
        )
        raise


if __name__ == "__main__":
    main()
