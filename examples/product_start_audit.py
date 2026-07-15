#!/usr/bin/env python
"""Audit product-state launches for LOMPS trajectories.

Two modes are useful for production checks:

``first-step``
    Build the exact first target from a product state, lift the optimizer seed
    to the requested ``D``, run CG plus LM polish, and record final tensor
    diagnostics.

``step2``
    Do the same first-step fit, then fit the second target once with the
    requested continuation tolerance and once with a relaxed tolerance. This
    exposes cases where an over-strict continuation tolerance spends a long
    time chasing the numerical floor.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from lomps.canonical import canonical_errors
from lomps.embedding import lift_left_canonical_seed, product_tensor
from lomps.evolution import fit_first_step_with_cg, fit_fixed_target, options
from lomps.fixed_target_cg import CGOptions
from lomps.optimizer import LMOptions, optimizer_right_fixed_point
from lomps.protocol import NONINTEGRABLE_ISING, LocalEvolutionProtocol
from lomps.rdm import block_rdm
from lomps.transfer import injectivity_diagnostics


def product_vector(name: str) -> np.ndarray:
    states = {
        "x": np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0),
        "y": np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0),
        "z": np.array([1.0, 0.0], dtype=np.complex128),
    }
    try:
        return states[name]
    except KeyError as error:
        raise ValueError(f"unknown product state {name!r}") from error


def parse_list(text: str, cast: type = float) -> tuple[Any, ...]:
    values = []
    for item in text.replace(" ", ",").split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return tuple(values)


def protocol(block_length: int, delta_t: float) -> LocalEvolutionProtocol:
    return replace(
        NONINTEGRABLE_ISING,
        name=f"nonintegrable_ising_L{block_length}",
        block_length=int(block_length),
        delta_t=float(delta_t),
    )


def tuned_options(
    *,
    accept_cost: float,
    rank_tolerance: float,
    fixed_point_solver: str,
    linear_solver: str,
    maximum_seconds: float,
) -> tuple[LMOptions, LMOptions]:
    primary, strict = options(
        accept_cost,
        rank_tolerance,
        fixed_point_solver,
        linear_solver,
    )
    return (
        replace(primary, maximum_seconds=maximum_seconds, verbose=False),
        replace(strict, maximum_seconds=maximum_seconds, verbose=False),
    )


def first_step_cg_controls(args: argparse.Namespace, accept_cost: float) -> CGOptions:
    return CGOptions(
        max_iterations=args.cg_iterations,
        gradient_tolerance=args.cg_gradient_tolerance,
        cost_tolerance=accept_cost,
        rank_tolerance=args.rank_tolerance,
        maximum_seconds=args.cg_seconds,
        fixed_point_solver=args.fixed_point_solver,
        precondition=not args.no_cg_precondition,
        verbose=args.verbose_cg,
    )


def fit_first_product_step(
    *,
    state: str,
    block_length: int,
    bond_dimension: int,
    delta_t: float,
    noise: float,
    embedding_seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = product_tensor(product_vector(state))
    local_protocol = protocol(block_length, delta_t)
    seed, seed_diag = lift_left_canonical_seed(
        source,
        bond_dimension,
        noise_amplitude=noise,
        seed=embedding_seed,
        diagnostic_block_length=block_length,
    )
    target = local_protocol.target_rdm(source)
    cg = first_step_cg_controls(args, args.first_step_accept_cost)
    lm_primary, lm_strict = tuned_options(
        accept_cost=args.first_step_accept_cost,
        rank_tolerance=args.rank_tolerance,
        fixed_point_solver=args.fixed_point_solver,
        linear_solver=args.linear_solver,
        maximum_seconds=args.polish_seconds,
    )

    started = time.perf_counter()
    cg_A, cg_result, cg_seconds = fit_first_step_with_cg(seed, target, cg)
    A1, polish_result, used_strict, polish_seconds = fit_fixed_target(
        cg_A,
        target,
        lm_primary,
        lm_strict,
        use_strict_retry=args.strict_retry,
    )
    total_seconds = time.perf_counter() - started

    r1, fixed_info = optimizer_right_fixed_point(A1, args.fixed_point_solver)
    rho1 = block_rdm(A1, block_length, r1)
    final_gap = float(injectivity_diagnostics(A1)["gap"])
    row = {
        "mode": "first-step",
        "state": state,
        "L": block_length,
        "D": bond_dimension,
        "delta_t": delta_t,
        "embedding_noise": noise,
        "embedding_seed": embedding_seed,
        "seed_gap": seed_diag.seed_transfer_gap,
        "seed_min_rfp_eig": seed_diag.seed_right_fixed_point_minimum_eigenvalue,
        "cg_cost": cg_result.cost,
        "cg_residual": cg_result.residual_norm,
        "cg_status": cg_result.status,
        "cg_evaluations": len(cg_result.history),
        "cg_seconds": cg_seconds,
        "polish_cost": polish_result.cost,
        "polish_residual": polish_result.residual_norm,
        "polish_status": polish_result.status,
        "polish_evaluations": len(polish_result.history),
        "polish_seconds": polish_seconds,
        "used_strict_retry": used_strict,
        "total_seconds": total_seconds,
        "rdm_frobenius_error": float(np.linalg.norm(rho1 - target)),
        "sqrt_2_cost": float(np.sqrt(max(2.0 * polish_result.cost, 0.0))),
        "final_gap": final_gap,
        "final_min_rfp_eig": float(fixed_info["min_eigenvalue"]),
        "left_canonical_error": canonical_errors(A1)["left_canonical_error"],
    }
    return A1, r1, row


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def iter_grid(args: argparse.Namespace) -> Iterable[tuple[str, int, int, float, float, int]]:
    for state in parse_list(args.states, str):
        for block_length in parse_list(args.block_lengths, int):
            for bond_dimension in parse_list(args.bond_dimensions, int):
                for delta_t in parse_list(args.delta_ts, float):
                    for noise in parse_list(args.embedding_noises, float):
                        for seed in parse_list(args.embedding_seeds, int):
                            yield state, block_length, bond_dimension, delta_t, noise, seed


def run_first_step(args: argparse.Namespace) -> None:
    for state, block_length, bond_dimension, delta_t, noise, seed in iter_grid(args):
        _, _, row = fit_first_product_step(
            state=state,
            block_length=block_length,
            bond_dimension=bond_dimension,
            delta_t=delta_t,
            noise=noise,
            embedding_seed=seed,
            args=args,
        )
        write_row(args.output_csv, row)
        print(
            "first-step "
            f"state={state} L={block_length} D={bond_dimension} dt={delta_t:g} "
            f"noise={noise:g} cost={row['polish_cost']:.2e} "
            f"rhoF={row['rdm_frobenius_error']:.2e} "
            f"gap={row['final_gap']:.2e} seconds={row['total_seconds']:.1f}",
            flush=True,
        )


def run_step2(args: argparse.Namespace) -> None:
    for state, block_length, bond_dimension, delta_t, noise, seed in iter_grid(args):
        A1, r1, first_row = fit_first_product_step(
            state=state,
            block_length=block_length,
            bond_dimension=bond_dimension,
            delta_t=delta_t,
            noise=noise,
            embedding_seed=seed,
            args=args,
        )
        local_protocol = protocol(block_length, delta_t)
        target2 = local_protocol.target_rdm(A1, r1)
        primary, strict = tuned_options(
            accept_cost=args.accept_cost,
            rank_tolerance=args.rank_tolerance,
            fixed_point_solver=args.fixed_point_solver,
            linear_solver=args.linear_solver,
            maximum_seconds=args.step2_seconds,
        )
        strict = replace(strict, maximum_seconds=args.strict_retry_seconds)
        strict_A, strict_result, used_strict, strict_seconds = fit_fixed_target(
            A1,
            target2,
            primary,
            strict,
            seed_fixed_point=r1,
            use_strict_retry=args.strict_retry,
        )
        relaxed_primary, relaxed_strict = tuned_options(
            accept_cost=args.relaxed_accept_cost,
            rank_tolerance=args.rank_tolerance,
            fixed_point_solver=args.fixed_point_solver,
            linear_solver=args.linear_solver,
            maximum_seconds=args.step2_seconds,
        )
        relaxed_A, relaxed_result, relaxed_used_strict, relaxed_seconds = fit_fixed_target(
            A1,
            target2,
            relaxed_primary,
            relaxed_strict,
            seed_fixed_point=r1,
            use_strict_retry=False,
        )
        strict_r, strict_info = optimizer_right_fixed_point(
            strict_A,
            args.fixed_point_solver,
        )
        relaxed_r, relaxed_info = optimizer_right_fixed_point(
            relaxed_A,
            args.fixed_point_solver,
        )
        row = {
            **first_row,
            "mode": "step2",
            "step2_accept_cost": args.accept_cost,
            "step2_strict_cost": strict_result.cost,
            "step2_strict_residual": strict_result.residual_norm,
            "step2_strict_status": strict_result.status,
            "step2_strict_evaluations": len(strict_result.history),
            "step2_used_strict_retry": used_strict,
            "step2_seconds": strict_seconds,
            "step2_gap": float(injectivity_diagnostics(strict_A)["gap"]),
            "step2_min_rfp_eig": float(strict_info["min_eigenvalue"]),
            "relaxed_accept_cost": args.relaxed_accept_cost,
            "relaxed_cost": relaxed_result.cost,
            "relaxed_residual": relaxed_result.residual_norm,
            "relaxed_status": relaxed_result.status,
            "relaxed_evaluations": len(relaxed_result.history),
            "relaxed_used_strict_retry": relaxed_used_strict,
            "relaxed_seconds": relaxed_seconds,
            "relaxed_gap": float(injectivity_diagnostics(relaxed_A)["gap"]),
            "relaxed_min_rfp_eig": float(relaxed_info["min_eigenvalue"]),
        }
        write_row(args.output_csv, row)
        print(
            "step2 "
            f"state={state} L={block_length} D={bond_dimension} dt={delta_t:g} "
            f"noise={noise:g} strict={strict_result.cost:.2e} "
            f"relaxed={relaxed_result.cost:.2e} seconds={strict_seconds:.1f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("first-step", "step2"), default="first-step")
    parser.add_argument("--states", default="x,y,z")
    parser.add_argument("--block-lengths", default="2,3,4")
    parser.add_argument("--bond-dimensions", default="4,6,8,12")
    parser.add_argument("--delta-ts", default="1e-3")
    parser.add_argument("--embedding-noises", default="1e-8,1e-6,1e-4")
    parser.add_argument("--embedding-seeds", default="104729")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/audits/product_start_audit.csv"),
    )
    parser.add_argument("--first-step-accept-cost", type=float, default=3e-16)
    parser.add_argument("--accept-cost", type=float, default=1e-15)
    parser.add_argument("--relaxed-accept-cost", type=float, default=1e-14)
    parser.add_argument("--rank-tolerance", type=float, default=1e-12)
    parser.add_argument("--fixed-point-solver", choices=("dense", "fast"), default="dense")
    parser.add_argument("--linear-solver", choices=("normal", "svd"), default="normal")
    parser.add_argument("--cg-seconds", type=float, default=50.0)
    parser.add_argument("--cg-iterations", type=int, default=40_000)
    parser.add_argument("--cg-gradient-tolerance", type=float, default=1e-11)
    parser.add_argument("--no-cg-precondition", action="store_true")
    parser.add_argument("--verbose-cg", action="store_true")
    parser.add_argument("--polish-seconds", type=float, default=30.0)
    parser.add_argument("--step2-seconds", type=float, default=20.0)
    parser.add_argument("--strict-retry-seconds", type=float, default=60.0)
    parser.add_argument(
        "--strict-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "first-step":
        run_first_step(args)
    else:
        run_step2(args)


if __name__ == "__main__":
    main()
