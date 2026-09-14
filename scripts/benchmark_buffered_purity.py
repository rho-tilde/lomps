#!/usr/bin/env python3
"""Smoke-test and benchmark exact-constraint buffered-purity descent.

The ``l2-smoke`` mode checks the mechanism on a small synthetic fibre.  The
``l4-trajectory`` mode uses successive tensors from the completed physical
``L=4,D=12`` non-integrable trajectory.  In both modes the secondary block is
four sites larger than the fitted block and no continuity penalty is used.
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
from lomps.canonical import canonical_errors, random_left_canonical
from lomps.optimizer import optimizer_right_fixed_point
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm


Array = np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    smoke = subparsers.add_parser("l2-smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--bond-dimension", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=2611)
    smoke.add_argument("--accept-cost", type=float, default=1e-12)
    smoke.add_argument("--iterations", type=int, default=5)
    smoke.add_argument("--initial-step", type=float, default=5e-2)

    l4 = subparsers.add_parser("l4-trajectory")
    l4.add_argument("--output-dir", type=Path, required=True)
    l4.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    l4.add_argument(
        "--sample-times",
        default="0.003,0.1,1.0,5.0,10.0,20.0",
        help="Comma-separated fitted-state times in physical units.",
    )
    l4.add_argument("--accept-cost", type=float, default=3e-16)
    l4.add_argument("--iterations", type=int, default=5)
    l4.add_argument("--initial-step", type=float, default=1e-2)
    l4.add_argument("--jacobian-workers", type=int, default=1)
    return parser.parse_args()


def trace_distance(rho: Array, sigma: Array) -> float:
    difference = 0.5 * ((rho - sigma) + (rho - sigma).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


def exact_metrics(
    A: Array,
    target: Array,
    block_length: int,
    buffer_sites: int = 4,
) -> dict[str, object]:
    r, fixed_info = optimizer_right_fixed_point(A, "dense")
    rho_primary = block_rdm(A, block_length, r)
    rho_buffered = block_rdm(A, block_length + buffer_sites, r)
    residual = rho_primary - target
    return {
        "primary_cost": 0.5 * float(np.vdot(residual, residual).real),
        "primary_trace_distance": trace_distance(rho_primary, target),
        "purity": density_matrix_purity(rho_buffered),
        "renyi2_entropy": -float(np.log(density_matrix_purity(rho_buffered))),
        "rho_primary": rho_primary,
        "rho_buffered": rho_buffered,
        "fixed_point_residual": float(fixed_info["residual"]),
    }


def controls(args: argparse.Namespace) -> BufferedPurityOptions:
    return BufferedPurityOptions(
        buffer_sites=4,
        primary_cost_tolerance=args.accept_cost,
        max_iterations=args.iterations,
        initial_step=args.initial_step,
        minimum_step=1e-10,
        reproject_primary=True,
        projection_max_iterations=8,
        projection_initial_damping=1e-10,
        projection_maximum_seconds=120.0,
        tangent_slice="grassmann",
        fixed_point_solver="dense",
        jacobian_response_solver="dense_lu",
        jacobian_workers=getattr(args, "jacobian_workers", 1),
        verbose=True,
    )


def serializable_history(result: object) -> list[dict[str, object]]:
    return [asdict(record) for record in result.history]


def run_l2_smoke(args: argparse.Namespace) -> None:
    A, _ = random_left_canonical(
        d=2, D=args.bond_dimension, seed=args.seed
    )
    target = block_rdm(A, 2)
    baseline = exact_metrics(A, target, 2)
    started = time.perf_counter()
    refined, result = minimize_buffered_purity(A, target, 2, controls(args))
    elapsed = time.perf_counter() - started
    final = exact_metrics(refined, target, 2)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "baseline.npy", A)
    np.save(args.output_dir / "refined.npy", refined)
    summary = {
        "mode": args.mode,
        "block_length": 2,
        "buffer_sites": 4,
        "extended_block_length": 6,
        "bond_dimension": args.bond_dimension,
        "seed": args.seed,
        "accept_cost": args.accept_cost,
        "elapsed_seconds": elapsed,
        "status": result.status,
        "accepted_purity_steps": result.accepted_steps,
        "baseline_primary_cost": baseline["primary_cost"],
        "final_primary_cost": final["primary_cost"],
        "baseline_purity": baseline["purity"],
        "final_purity": final["purity"],
        "relative_purity_drop": (
            (baseline["purity"] - final["purity"]) / baseline["purity"]
        ),
        "left_canonical_error": canonical_errors(refined)[
            "left_canonical_error"
        ],
        "history": serializable_history(result),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


class L4Trajectory:
    def __init__(self, data_dir: Path):
        self.initial = np.asarray(
            np.load(data_dir / "nonintegrable_d12_t0p001.npy"),
            dtype=np.complex128,
        )
        root = data_dir / "nonintegrable_d12_trajectory"
        self.states = np.load(root / "trajectory_states.npy", mmap_mode="r")
        self.times = np.load(root / "trajectory_times.npy", mmap_mode="r")

    def state_at_step(self, physical_step: int) -> Array:
        """Return the saved tensor at ``t = physical_step * 0.001``."""

        if physical_step == 1:
            return self.initial.copy()
        if physical_step < 1 or physical_step > 20_000:
            raise ValueError("physical_step must lie in [1, 20000]")
        return np.asarray(self.states[physical_step - 2], dtype=np.complex128)


def run_l4_trajectory(args: argparse.Namespace) -> None:
    trajectory = L4Trajectory(args.data_dir)
    sample_times = [
        float(value) for value in args.sample_times.split(",") if value.strip()
    ]
    if not sample_times:
        raise ValueError("at least one sample time is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = replace(
        NONINTEGRABLE_ISING,
        block_length=4,
        target_source_fixed_point_solver="dense",
    )
    option_values = controls(args)
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for sample_time in sample_times:
        physical_step = int(round(sample_time / protocol.delta_t))
        represented_time = physical_step * protocol.delta_t
        if abs(represented_time - sample_time) > 1e-12:
            raise ValueError(f"sample time {sample_time} is off the dt grid")
        if physical_step < 2:
            raise ValueError("L4 benchmark sample times must be at least 0.002")
        source = trajectory.state_at_step(physical_step - 1)
        candidate = trajectory.state_at_step(physical_step)

        target_started = time.perf_counter()
        target = protocol.target_rdm(source)
        target_seconds = time.perf_counter() - target_started
        baseline = exact_metrics(candidate, target, 4)
        if baseline["primary_cost"] > args.accept_cost:
            raise RuntimeError(
                f"baseline at t={sample_time} is not feasible: "
                f"{baseline['primary_cost']:.6e} > {args.accept_cost:.6e}"
            )

        started = time.perf_counter()
        refined, result = minimize_buffered_purity(
            candidate, target, 4, option_values
        )
        optimizer_seconds = time.perf_counter() - started
        final = exact_metrics(refined, target, 4)
        if final["primary_cost"] > args.accept_cost:
            raise RuntimeError(
                f"refined point at t={sample_time} violated the primary cost"
            )

        label = f"t{sample_time:.3f}".replace(".", "p")
        np.save(args.output_dir / f"refined_{label}.npy", refined)
        baseline_next_target = protocol.target_rdm(candidate)
        refined_next_target = protocol.target_rdm(refined)
        next_target_difference = baseline_next_target - refined_next_target
        relative_drop = (baseline["purity"] - final["purity"]) / baseline[
            "purity"
        ]
        row = {
            "time": sample_time,
            "physical_step": physical_step,
            "baseline_primary_cost": baseline["primary_cost"],
            "refined_primary_cost": final["primary_cost"],
            "baseline_primary_trace_distance": baseline[
                "primary_trace_distance"
            ],
            "refined_primary_trace_distance": final["primary_trace_distance"],
            "baseline_purity_rho8": baseline["purity"],
            "refined_purity_rho8": final["purity"],
            "relative_purity_drop": relative_drop,
            "baseline_renyi2_rho8": baseline["renyi2_entropy"],
            "refined_renyi2_rho8": final["renyi2_entropy"],
            "rho4_baseline_refined_trace_distance": trace_distance(
                baseline["rho_primary"], final["rho_primary"]
            ),
            "rho8_baseline_refined_trace_distance": trace_distance(
                baseline["rho_buffered"], final["rho_buffered"]
            ),
            "next_target_trace_distance": trace_distance(
                baseline_next_target, refined_next_target
            ),
            "next_target_half_squared_frobenius": 0.5
            * float(np.vdot(next_target_difference, next_target_difference).real),
            "accepted_purity_steps": result.accepted_steps,
            "status": result.status,
            "target_seconds": target_seconds,
            "optimizer_seconds": optimizer_seconds,
            "projection_evaluations": sum(
                record.projection_evaluations for record in result.history
            ),
            "initial_visible_rank": (
                result.history[0].visible_rank if result.history else -1
            ),
            "tangent_dimension": (
                result.history[0].tangent_dimension if result.history else -1
            ),
            "left_canonical_error": canonical_errors(refined)[
                "left_canonical_error"
            ],
        }
        rows.append(row)
        details.append({**row, "history": serializable_history(result)})
        print(json.dumps(row, sort_keys=True), flush=True)

    with (args.output_dir / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mode": args.mode,
        "block_length": 4,
        "buffer_sites": 4,
        "extended_block_length": 8,
        "bond_dimension": 12,
        "protocol": protocol.name,
        "delta_t": protocol.delta_t,
        "accept_cost": args.accept_cost,
        "options": asdict(option_values),
        "sample_count": len(rows),
        "all_primary_costs_accepted": all(
            float(row["refined_primary_cost"]) <= args.accept_cost for row in rows
        ),
        "all_purities_reduced": all(
            float(row["relative_purity_drop"]) > 0 for row in rows
        ),
        "mean_relative_purity_drop": float(
            np.mean([float(row["relative_purity_drop"]) for row in rows])
        ),
        "total_optimizer_seconds": float(
            sum(float(row["optimizer_seconds"]) for row in rows)
        ),
        "samples": details,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.accept_cost <= 0:
        raise ValueError("--accept-cost must be positive")
    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative")
    if args.mode == "l2-smoke":
        run_l2_smoke(args)
    else:
        run_l4_trajectory(args)


if __name__ == "__main__":
    main()
