#!/usr/bin/env python3
"""Run a short sequential L=4 buffered-purity trajectory benchmark.

The branch starts from a saved ``L=4,D=12`` physical state.  Every subsequent
step first performs the ordinary accurate LOMPS fit and then minimizes the
purity of ``rho_8`` while retaining the same hard ``rho_4`` cost.  There is no
continuity penalty.  Saved baseline L=4 and higher-L L=5 RDMs are used only as
diagnostics; they do not enter either optimization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from lomps.buffered_purity import (
    BufferedPurityOptions,
    density_matrix_purity,
    minimize_buffered_purity,
)
from lomps.canonical import canonical_errors
from lomps.optimizer import LMOptions, optimize_tensor, optimizer_right_fixed_point
from lomps.predictor import trajectory_secant_predictor
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm


Array = np.ndarray


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--start-time", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--accept-cost", type=float, default=3e-16)
    parser.add_argument("--purity-iterations", type=int, default=5)
    parser.add_argument("--purity-step", type=float, default=1e-2)
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
        return np.asarray(self.l4_states[physical_step - 2], dtype=np.complex128)

    def l5(self, time_value: float) -> Array:
        if time_value <= float(self.l5_times_d20[-1]):
            times, states = self.l5_times_d20, self.l5_states_d20
        else:
            times, states = self.l5_times_d21, self.l5_states_d21
        index = int(round((time_value - float(times[0])) / 1e-3))
        if index < 0 or index >= len(times):
            raise ValueError(f"no L=5 reference at t={time_value}")
        if abs(float(times[index]) - time_value) > 1e-12:
            raise ValueError(f"L=5 reference grid mismatch at t={time_value}")
        return np.asarray(states[index], dtype=np.complex128)


def dense_primary_cost(A: Array, target: Array) -> float:
    r, _ = optimizer_right_fixed_point(A, "dense")
    residual = block_rdm(A, 4, r) - target
    return 0.5 * float(np.vdot(residual, residual).real)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    protocol = replace(
        NONINTEGRABLE_ISING,
        block_length=4,
        target_source_fixed_point_solver="dense",
    )
    start_step = int(round(args.start_time / protocol.delta_t))
    if abs(start_step * protocol.delta_t - args.start_time) > 1e-12:
        raise ValueError("--start-time must lie on the dt grid")
    references = References(args.data_dir)
    previous = references.l4(start_step - 1)
    current = references.l4(start_step)

    primary_options = LMOptions(
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
        jacobian_workers=1,
        verbose=False,
    )
    purity_options = BufferedPurityOptions(
        buffer_sites=4,
        primary_cost_tolerance=args.accept_cost,
        max_iterations=args.purity_iterations,
        initial_step=args.purity_step,
        minimum_step=1e-10,
        reproject_primary=True,
        projection_max_iterations=8,
        projection_initial_damping=1e-10,
        projection_maximum_seconds=120.0,
        tangent_slice="grassmann",
        fixed_point_solver="dense",
        jacobian_response_solver="dense_lu",
        jacobian_workers=1,
        verbose=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    branch_states = np.empty(
        (args.steps + 1, *current.shape), dtype=np.complex128
    )
    branch_times = args.start_time + protocol.delta_t * np.arange(args.steps + 1)
    branch_states[0] = current
    rows: list[dict[str, object]] = []
    started_all = time.perf_counter()

    for local_step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        target = protocol.target_rdm(current)
        seed = trajectory_secant_predictor(previous, current)
        fit_started = time.perf_counter()
        fitted, fit_result = optimize_tensor(seed, target, 4, primary_options)
        fit_seconds = time.perf_counter() - fit_started
        fitted_cost = dense_primary_cost(fitted, target)
        if fitted_cost > args.accept_cost:
            raise RuntimeError(
                f"ordinary fit failed at local step {local_step}: "
                f"{fitted_cost:.6e}"
            )
        fitted_r, _ = optimizer_right_fixed_point(fitted, "dense")
        fitted_purity = density_matrix_purity(block_rdm(fitted, 8, fitted_r))

        purity_started = time.perf_counter()
        refined, purity_result = minimize_buffered_purity(
            fitted, target, 4, purity_options
        )
        purity_seconds = time.perf_counter() - purity_started
        refined_cost = dense_primary_cost(refined, target)
        if refined_cost > args.accept_cost:
            raise RuntimeError(
                f"purity refinement failed at local step {local_step}: "
                f"{refined_cost:.6e}"
            )
        refined_r, _ = optimizer_right_fixed_point(refined, "dense")
        refined_rho4 = block_rdm(refined, 4, refined_r)
        refined_purity = density_matrix_purity(block_rdm(refined, 8, refined_r))

        physical_step = start_step + local_step
        time_value = physical_step * protocol.delta_t
        baseline = references.l4(physical_step)
        reference = references.l5(time_value)
        baseline_rho4 = block_rdm(baseline, 4)
        reference_rho4 = block_rdm(reference, 4)
        row = {
            "local_step": local_step,
            "time": time_value,
            "ordinary_fit_cost": fitted_cost,
            "refined_fit_cost": refined_cost,
            "ordinary_fit_status": fit_result.status,
            "ordinary_fit_evaluations": len(fit_result.history),
            "purity_status": purity_result.status,
            "purity_steps": purity_result.accepted_steps,
            "purity_before": fitted_purity,
            "purity_after": refined_purity,
            "relative_purity_drop": (fitted_purity - refined_purity)
            / fitted_purity,
            "branch_to_baseline_rho4_trace_distance": trace_distance(
                refined_rho4, baseline_rho4
            ),
            "branch_to_l5_rho4_trace_distance": trace_distance(
                refined_rho4, reference_rho4
            ),
            "baseline_to_l5_rho4_trace_distance": trace_distance(
                baseline_rho4, reference_rho4
            ),
            "fit_seconds": fit_seconds,
            "purity_seconds": purity_seconds,
            "step_seconds": time.perf_counter() - step_started,
            "left_canonical_error": canonical_errors(refined)[
                "left_canonical_error"
            ],
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        previous, current = current, refined
        branch_states[local_step] = current

    np.save(args.output_dir / "states.npy", branch_states)
    np.save(args.output_dir / "times.npy", branch_times)
    with (args.output_dir / "steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed",
        "protocol": protocol.name,
        "block_length": 4,
        "buffer_sites": 4,
        "extended_block_length": 8,
        "bond_dimension": 12,
        "start_time": args.start_time,
        "final_time": float(branch_times[-1]),
        "steps": args.steps,
        "accept_cost": args.accept_cost,
        "primary_options": asdict(primary_options),
        "purity_options": asdict(purity_options),
        "all_costs_accepted": all(
            float(row["refined_fit_cost"]) <= args.accept_cost for row in rows
        ),
        "all_purities_reduced": all(
            float(row["relative_purity_drop"]) > 0 for row in rows
        ),
        "initial_baseline_to_l5_trace_distance": float(
            rows[0]["baseline_to_l5_rho4_trace_distance"]
        ),
        "final_branch_to_l5_trace_distance": float(
            rows[-1]["branch_to_l5_rho4_trace_distance"]
        ),
        "final_baseline_to_l5_trace_distance": float(
            rows[-1]["baseline_to_l5_rho4_trace_distance"]
        ),
        "total_seconds": time.perf_counter() - started_all,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
