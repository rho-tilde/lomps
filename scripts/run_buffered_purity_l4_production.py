#!/usr/bin/env python3
"""Run a portable, resumable L=4 control or buffered-purity branch.

The run starts from two explicit D=12 tensors at ``t-dt`` and ``t``.  Both
branches first refit the anchor to a strict internal rho4 target.  The
``purity`` branch then minimizes ``Tr(rho8**2)`` at the anchor and after every
physical step.  A relative-purity plateau is a recorded stall rather than
formal projected-gradient convergence.  Deep searches may nevertheless be
accepted as numerical stationarity when the line search reaches machine-scale
purity changes after a substantial number of accepted fibre updates; that
distinction is preserved in every output row.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Literal

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
Mode = Literal["control", "purity"]
CSV_FIELDS = [
    "local_step",
    "time",
    "mode",
    "primary_cost",
    "ordinary_fit_status",
    "ordinary_fit_evaluations",
    "purity_search_kind",
    "purity_converged",
    "purity_stationarity",
    "purity_status",
    "purity_iterations",
    "purity_before",
    "purity_after",
    "relative_purity_drop",
    "terminal_null_gradient_norm",
    "terminal_relative_purity_drop",
    "visible_rank",
    "tangent_dimension",
    "fit_seconds",
    "purity_seconds",
    "step_seconds",
    "left_canonical_error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_purity_history(path: Path, result: BufferedPurityResult) -> None:
    """Store a compact, complete inner-iteration ledger for one fibre search."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    records = result.history
    names = tuple(records[0].__dataclass_fields__) if records else ()
    payload: dict[str, Array] = {
        name: np.asarray([getattr(record, name) for record in records])
        for name in names
    }
    payload.update(
        status=np.asarray(result.status),
        accepted_steps=np.asarray(result.accepted_steps, dtype=np.int64),
        initial_primary_cost=np.asarray(result.initial_primary_cost),
        final_primary_cost=np.asarray(result.primary_cost),
        initial_purity=np.asarray(result.initial_purity),
        final_purity=np.asarray(result.purity),
        final_null_gradient_norm=np.asarray(result.final_null_gradient_norm),
        final_relative_purity_drop=np.asarray(
            result.final_relative_purity_drop
        ),
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-A", type=Path, required=True)
    parser.add_argument("--anchor-A", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "purity"), required=True)
    parser.add_argument("--start-time", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--accept-cost", type=float, default=3e-16)
    parser.add_argument("--fit-cost-target", type=float, default=1e-18)
    parser.add_argument("--purity-step", type=float, default=50.0)
    parser.add_argument("--purity-max-iterations", type=int, default=2000)
    parser.add_argument("--purity-tracking-iterations", type=int, default=0)
    parser.add_argument("--purity-full-every", type=int, default=1)
    parser.add_argument("--purity-gradient-tolerance", type=float, default=2e-8)
    parser.add_argument("--purity-relative-tolerance", type=float, default=1e-12)
    parser.add_argument("--purity-relative-patience", type=int, default=10)
    parser.add_argument("--purity-minimum-full-iterations", type=int, default=200)
    parser.add_argument("--jacobian-workers", type=int, default=1)
    parser.add_argument("--pause-after-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-purity", action="store_true")
    return parser.parse_args()


def load_tensor(path: Path) -> Array:
    tensor = np.asarray(np.load(path), dtype=np.complex128)
    if tensor.shape != (2, 12, 12):
        raise ValueError(f"expected tensor shape (2, 12, 12), got {tensor.shape}")
    return tensor


def exact_rdms(A: Array) -> tuple[Array, Array]:
    right, _ = optimizer_right_fixed_point(A, "dense")
    return block_rdm(A, 4, right), block_rdm(A, 8, right)


def exact_primary_cost(A: Array, target: Array) -> float:
    rho4, _ = exact_rdms(A)
    residual = rho4 - target
    return 0.5 * float(np.vdot(residual, residual).real)


def fit_options(args: argparse.Namespace) -> LMOptions:
    return LMOptions(
        max_iterations=80,
        gradient_tolerance=1e-11,
        cost_tolerance=args.fit_cost_target,
        rank_tolerance=1e-10,
        tangent_slice="grassmann",
        initial_damping=1e-10,
        maximum_seconds=600.0,
        fixed_point_solver="dense",
        linear_solver="normal",
        rdm_vectorization="hermitian",
        jacobian_response_solver="dense_lu",
        jacobian_workers=args.jacobian_workers,
        verbose=False,
    )


def purity_options(args: argparse.Namespace) -> BufferedPurityOptions:
    return BufferedPurityOptions(
        buffer_sites=4,
        primary_cost_tolerance=args.accept_cost,
        projection_cost_tolerance=args.fit_cost_target,
        max_iterations=args.purity_max_iterations,
        initial_step=args.purity_step,
        direction_scaling="gradient",
        minimum_step=1e-10,
        null_gradient_tolerance=args.purity_gradient_tolerance,
        relative_purity_tolerance=args.purity_relative_tolerance,
        relative_purity_patience=args.purity_relative_patience,
        relative_purity_is_convergence=False,
        reproject_primary=True,
        projection_max_iterations=12,
        projection_initial_damping=1e-10,
        projection_maximum_seconds=600.0,
        tangent_slice="grassmann",
        fixed_point_solver="dense",
        jacobian_response_solver="dense_lu",
        jacobian_workers=args.jacobian_workers,
        verbose=not args.quiet_purity,
    )


def require_primary_cost(cost: float, target: float, label: str) -> None:
    if not np.isfinite(cost) or cost > target:
        raise RuntimeError(f"{label}: exact C4={cost:.6e} exceeds {target:.6e}")


def classify_full_purity_search(
    result: BufferedPurityResult,
    *,
    minimum_iterations: int,
    relative_tolerance: float,
    label: str,
) -> str:
    """Require and classify projected-gradient or numerical stationarity."""

    if result.status == "null_gradient_tolerance":
        return "projected_gradient"
    numerical_floor = (
        result.status in {"line_search_failed", "relative_purity_stall"}
        and result.accepted_steps >= minimum_iterations
        and np.isfinite(result.final_relative_purity_drop)
        and 0.0 <= result.final_relative_purity_drop <= relative_tolerance
    )
    if numerical_floor:
        return "numerical_line_search_floor"
    raise RuntimeError(
        f"{label}: purity search did not reach fibre stationarity; "
        f"status={result.status}, iterations={result.accepted_steps}, "
        f"|g_fibre|={result.final_null_gradient_norm:.6e}, "
        f"last_relative_drop={result.final_relative_purity_drop:.6e}"
    )


def require_purity_tracking(
    result: BufferedPurityResult,
    expected_iterations: int,
    label: str,
) -> None:
    if result.status == "null_gradient_tolerance":
        return
    if (
        result.status == "maximum_iterations"
        and result.accepted_steps == expected_iterations
    ):
        return
    raise RuntimeError(
        f"{label}: purity tracking failed; status={result.status}, "
        f"accepted={result.accepted_steps}/{expected_iterations}, "
        f"|g_fibre|={result.final_null_gradient_norm:.6e}"
    )


def purity_rank(result: BufferedPurityResult) -> tuple[int | None, int | None]:
    if not result.history:
        return None, None
    record = result.history[-1]
    return record.visible_rank, record.tangent_dimension


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
    if not path.exists():
        if completed_steps:
            raise RuntimeError("checkpoint has steps but steps.csv is missing")
        return
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    durable = [row for row in rows if int(row["local_step"]) <= completed_steps]
    observed = [int(row["local_step"]) for row in durable]
    expected = list(range(1, completed_steps + 1))
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


def update_manifest(path: Path, manifest: dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at"] = utc_now()
    atomic_json(path, manifest)


def save_state(states_dir: Path, local_step: int, time_value: float, A: Array) -> None:
    np.save(states_dir / f"step_{local_step:06d}_t{time_value:.3f}.npy", A)


def refine_purity(
    A: Array,
    target: Array,
    options: BufferedPurityOptions,
    label: str,
) -> tuple[Array, BufferedPurityResult, float]:
    started = time.perf_counter()
    refined, result = minimize_buffered_purity(A, target, 4, options)
    elapsed = time.perf_counter() - started
    return refined, result, elapsed


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.fit_cost_target <= 0 or args.fit_cost_target > args.accept_cost:
        raise ValueError("--fit-cost-target must lie in (0, --accept-cost]")
    if args.purity_relative_patience < 1:
        raise ValueError("--purity-relative-patience must be positive")
    if args.purity_tracking_iterations < 0:
        raise ValueError("--purity-tracking-iterations cannot be negative")
    if args.purity_full_every < 1:
        raise ValueError("--purity-full-every must be positive")
    if args.purity_minimum_full_iterations < 1:
        raise ValueError("--purity-minimum-full-iterations must be positive")
    if args.jacobian_workers < 1:
        raise ValueError("--jacobian-workers must be positive")

    protocol = replace(
        NONINTEGRABLE_ISING,
        block_length=4,
        target_source_fixed_point_solver="dense",
    )
    start_step = int(round(args.start_time / protocol.delta_t))
    if abs(start_step * protocol.delta_t - args.start_time) > 1e-12:
        raise ValueError("--start-time must lie on the dt grid")

    previous_path = args.previous_A.resolve()
    anchor_path = args.anchor_A.resolve()
    previous_sha256 = sha256(previous_path)
    anchor_sha256 = sha256(anchor_path)
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    checkpoint_path = output / "checkpoint.npz"
    steps_path = output / "steps.csv"
    states_dir = output / "states"
    purity_history_dir = output / "purity_history"
    f_options = fit_options(args)
    p_options = purity_options(args)
    tracking_options = replace(
        p_options,
        max_iterations=args.purity_tracking_iterations,
        relative_purity_patience=max(
            p_options.relative_purity_patience,
            args.purity_tracking_iterations + 1,
        ),
    )
    config = {
        "protocol": "nonintegrable_ising_yplus",
        "mode": args.mode,
        "block_length": 4,
        "buffer_sites": 4,
        "extended_block_length": 8,
        "bond_dimension": 12,
        "delta_t": protocol.delta_t,
        "start_time": args.start_time,
        "target_steps": args.steps,
        "final_time": args.start_time + args.steps * protocol.delta_t,
        "previous_A": str(previous_path),
        "previous_A_sha256": previous_sha256,
        "anchor_A": str(anchor_path),
        "anchor_A_sha256": anchor_sha256,
        "accept_cost": args.accept_cost,
        "fit_cost_target": args.fit_cost_target,
        "primary_options": asdict(f_options),
        "purity_options": asdict(p_options) if args.mode == "purity" else None,
        "purity_tracking_iterations": args.purity_tracking_iterations,
        "purity_full_every": args.purity_full_every,
        "purity_minimum_full_iterations": args.purity_minimum_full_iterations,
        "no_continuity_penalty": True,
    }
    run_started = time.monotonic()

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
        states_dir.mkdir(exist_ok=True)
        purity_history_dir.mkdir(exist_ok=True)
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
        purity_history_dir.mkdir()
        completed_steps = 0
        current_time = args.start_time
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "pid": os.getpid(),
            "config": config,
            "completed_steps": 0,
            "current_time": current_time,
            "current_stage": "anchor_fit",
        }
        atomic_json(manifest_path, manifest)
        try:
            source_previous = load_tensor(previous_path)
            anchor_seed = load_tensor(anchor_path)
            anchor_target = protocol.target_rdm(source_previous)
            anchor_fit_started = time.perf_counter()
            fitted_anchor, fit_result = optimize_tensor(
                anchor_seed, anchor_target, 4, f_options
            )
            anchor_fit_seconds = time.perf_counter() - anchor_fit_started
            anchor_fit_cost = exact_primary_cost(fitted_anchor, anchor_target)
            require_primary_cost(
                anchor_fit_cost, args.fit_cost_target, "anchor fit"
            )
            _, fitted_anchor_rho8 = exact_rdms(fitted_anchor)
            anchor_purity_before = density_matrix_purity(fitted_anchor_rho8)
            anchor_purity_result: BufferedPurityResult | None = None
            anchor_purity_stationarity = "not_run"
            anchor_purity_seconds = 0.0
            current = fitted_anchor
            if args.mode == "purity":
                update_manifest(
                    manifest_path,
                    manifest,
                    current_stage="anchor_purity_refinement",
                    anchor_fit_cost=anchor_fit_cost,
                )
                current, anchor_purity_result, anchor_purity_seconds = (
                    refine_purity(fitted_anchor, anchor_target, p_options, "anchor")
                )
                atomic_purity_history(
                    purity_history_dir / "step_000000.npz",
                    anchor_purity_result,
                )
                anchor_purity_stationarity = classify_full_purity_search(
                    anchor_purity_result,
                    minimum_iterations=args.purity_minimum_full_iterations,
                    relative_tolerance=args.purity_relative_tolerance,
                    label="anchor",
                )
                require_primary_cost(
                    anchor_purity_result.primary_cost,
                    p_options.primary_cost_tolerance,
                    "anchor",
                )
            anchor_cost = exact_primary_cost(current, anchor_target)
            require_primary_cost(anchor_cost, args.accept_cost, "anchor")
            _, anchor_rho8 = exact_rdms(current)
            anchor_purity_after = density_matrix_purity(anchor_rho8)
            previous = source_previous
            save_state(states_dir, 0, current_time, current)
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
                current_stage="anchor_checkpointed",
                anchor={
                    "ordinary_fit_cost": anchor_fit_cost,
                    "ordinary_fit_status": fit_result.status,
                    "ordinary_fit_evaluations": len(fit_result.history),
                    "primary_cost": anchor_cost,
                    "purity_before": anchor_purity_before,
                    "purity_after": anchor_purity_after,
                    "purity_status": (
                        None
                        if anchor_purity_result is None
                        else anchor_purity_result.status
                    ),
                    "purity_stationarity": anchor_purity_stationarity,
                    "purity_iterations": (
                        0
                        if anchor_purity_result is None
                        else anchor_purity_result.accepted_steps
                    ),
                    "terminal_null_gradient_norm": (
                        None
                        if anchor_purity_result is None
                        else anchor_purity_result.final_null_gradient_norm
                    ),
                    "fit_seconds": anchor_fit_seconds,
                    "purity_seconds": anchor_purity_seconds,
                },
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

    try:
        for local_step in range(completed_steps + 1, args.steps + 1):
            if args.pause_after_seconds > 0 and (
                time.monotonic() - run_started >= args.pause_after_seconds
            ):
                update_manifest(
                    manifest_path,
                    manifest,
                    status="paused_by_walltime",
                    current_stage="paused",
                    completed_steps=completed_steps,
                    current_time=current_time,
                    paused_at=utc_now(),
                )
                return

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
            seed = (
                current.copy()
                if local_step == 1
                else trajectory_secant_predictor(previous, current)
            )
            fit_started = time.perf_counter()
            fitted, fit_result = optimize_tensor(seed, target, 4, f_options)
            fit_seconds = time.perf_counter() - fit_started
            fitted_cost = exact_primary_cost(fitted, target)
            require_primary_cost(
                fitted_cost, args.fit_cost_target, f"step {local_step} fit"
            )
            _, fitted_rho8 = exact_rdms(fitted)
            purity_before = density_matrix_purity(fitted_rho8)

            refined = fitted
            purity_result: BufferedPurityResult | None = None
            purity_seconds = 0.0
            purity_search_kind = "not_run"
            purity_converged = False
            purity_stationarity = "not_run"
            if args.mode == "purity":
                full_search = (
                    args.purity_tracking_iterations == 0
                    or local_step % args.purity_full_every == 0
                )
                purity_search_kind = "full" if full_search else "tracking"
                active_purity_options = (
                    p_options if full_search else tracking_options
                )
                update_manifest(
                    manifest_path,
                    manifest,
                    current_stage="purity_refinement",
                    ordinary_fit_cost=fitted_cost,
                    ordinary_fit_status=fit_result.status,
                    ordinary_fit_evaluations=len(fit_result.history),
                    purity_search_kind=purity_search_kind,
                )
                refined, purity_result, purity_seconds = refine_purity(
                    fitted,
                    target,
                    active_purity_options,
                    f"step {local_step}",
                )
                atomic_purity_history(
                    purity_history_dir / f"step_{local_step:06d}.npz",
                    purity_result,
                )
                if full_search:
                    purity_stationarity = classify_full_purity_search(
                        purity_result,
                        minimum_iterations=args.purity_minimum_full_iterations,
                        relative_tolerance=args.purity_relative_tolerance,
                        label=f"step {local_step}",
                    )
                else:
                    require_purity_tracking(
                        purity_result,
                        args.purity_tracking_iterations,
                        f"step {local_step}",
                    )
                    purity_stationarity = (
                        "projected_gradient"
                        if purity_result.status == "null_gradient_tolerance"
                        else "tracking_budget"
                    )
                purity_converged = (
                    purity_result.status == "null_gradient_tolerance"
                )
                require_primary_cost(
                    purity_result.primary_cost,
                    active_purity_options.primary_cost_tolerance,
                    f"step {local_step}",
                )

            refined_rho4, refined_rho8 = exact_rdms(refined)
            residual = refined_rho4 - target
            refined_cost = 0.5 * float(np.vdot(residual, residual).real)
            require_primary_cost(refined_cost, args.accept_cost, f"step {local_step}")
            purity_after = density_matrix_purity(refined_rho8)
            visible_rank, tangent_dimension = (
                (None, None)
                if purity_result is None
                else purity_rank(purity_result)
            )
            row = {
                "local_step": local_step,
                "time": next_time,
                "mode": args.mode,
                "primary_cost": refined_cost,
                "ordinary_fit_status": fit_result.status,
                "ordinary_fit_evaluations": len(fit_result.history),
                "purity_search_kind": purity_search_kind,
                "purity_converged": purity_converged,
                "purity_stationarity": purity_stationarity,
                "purity_status": (
                    "not_run" if purity_result is None else purity_result.status
                ),
                "purity_iterations": (
                    0 if purity_result is None else purity_result.accepted_steps
                ),
                "purity_before": purity_before,
                "purity_after": purity_after,
                "relative_purity_drop": (
                    (purity_before - purity_after) / purity_before
                ),
                "terminal_null_gradient_norm": (
                    None
                    if purity_result is None
                    else purity_result.final_null_gradient_norm
                ),
                "terminal_relative_purity_drop": (
                    None
                    if purity_result is None
                    else purity_result.final_relative_purity_drop
                ),
                "visible_rank": visible_rank,
                "tangent_dimension": tangent_dimension,
                "fit_seconds": fit_seconds,
                "purity_seconds": purity_seconds,
                "step_seconds": time.perf_counter() - step_started,
                "left_canonical_error": canonical_errors(refined)[
                    "left_canonical_error"
                ],
            }
            append_row(steps_path, row)
            save_state(states_dir, local_step, next_time, refined)
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
