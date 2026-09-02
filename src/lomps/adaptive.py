"""Run a LOMPS trajectory with monotone bond-dimension promotion.

Each bond dimension is handled by an ordinary ``lomps-run`` child process.
When a child exhausts its optimizer without accepting the next update, the
last accepted tensor remains the exact source for a new, higher-D segment.
The best rejected tensor is carried separately as the optimizer seed, so the
new segment preserves optimization progress without changing its physical
target.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import scipy


PROMOTABLE_STATUSES = {
    "failed_fixed_target_multistart",
    "promote_requested_by_runtime",
}
RESUMABLE_STATUSES = {
    "paused_by_PAUSE_file",
    "paused_by_step_limit",
    "paused_by_run_time_limit",
    "paused_by_keyboard_interrupt",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_provenance() -> dict[str, Any]:
    """Fingerprint the exact local LOMPS implementation and runtime."""

    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(package.glob("*.py"))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "source_tree_sha256": digest.hexdigest(),
        "source_files": [path.name for path in files],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def parse_bond_dimensions(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("at least one bond dimension is required")
    if any(value < 1 for value in values):
        raise ValueError("bond dimensions must be positive")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("bond dimensions must be strictly increasing")
    return values


def optimizer_policy(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return the optional non-default optimizer policy for the manifest."""

    backend = getattr(args, "optimizer", "dense-lm")
    matrix_free_from = getattr(args, "matrix_free_from_bond_dimension", 0)
    if backend == "dense-lm" and matrix_free_from <= 0:
        return None
    policy = {
        "backend": backend,
        "krylov_solver": args.matrix_free_krylov_solver,
        "krylov_initial_iterations": (
            args.matrix_free_krylov_initial_iterations
        ),
        "krylov_max_iterations": args.matrix_free_krylov_max_iterations,
        "krylov_preconditioner": args.matrix_free_krylov_preconditioner,
        "krylov_relative_tolerance": (
            args.matrix_free_krylov_relative_tolerance
        ),
        "krylov_minimum_relative_tolerance": (
            args.matrix_free_krylov_minimum_relative_tolerance
        ),
        "adaptive_krylov_tolerance": (
            args.matrix_free_adaptive_krylov_tolerance
        ),
        "recycle_krylov_solution": (
            args.matrix_free_recycle_krylov_solution
        ),
        "lsmr_condition_limit": args.matrix_free_lsmr_condition_limit,
        "fixed_point_response_solver": (
            args.matrix_free_fixed_point_response_solver
        ),
        "fixed_point_response_max_mb": (
            args.matrix_free_fixed_point_response_max_mb
        ),
        "adjoint_rtol": args.matrix_free_adjoint_rtol,
        "jvp_fixed_point_rtol": args.matrix_free_jvp_rtol,
        "verbose": args.matrix_free_verbose,
    }
    if matrix_free_from > 0:
        policy["matrix_free_from_bond_dimension"] = matrix_free_from
    if getattr(args, "matrix_free_dense_rescue", False):
        policy["dense_rescue"] = {
            "maximum_seconds": getattr(
                args, "matrix_free_dense_rescue_seconds", 3600.0
            ),
            "iteration_cap": getattr(
                args, "matrix_free_dense_rescue_iteration_cap", 0
            ),
        }
    return policy


def adaptive_policy(args: argparse.Namespace) -> dict[str, Any]:
    """Return every adaptive control that must remain fixed on resume."""

    policy = {
        "initial_key": args.initial_key,
        "initial_layout": args.initial_layout,
        "protocol": args.protocol,
        "g": args.g,
        "h": args.h,
        "J": args.J,
        "trotter_order": args.trotter_order,
        "symmetric_transverse": args.symmetric_transverse,
        "accept_cost": args.accept_cost,
        "initial_first_step_accept_cost": args.initial_first_step_accept_cost,
        "handoff_accept_cost": args.handoff_accept_cost,
        "initial_seed_lift_noise_amplitude": (
            args.initial_seed_lift_noise_amplitude
        ),
        "embedding_noise_amplitude": args.embedding_noise_amplitude,
        "embedding_seed": args.embedding_seed,
        "first_step_optimizer": args.first_step_optimizer,
        "initial_first_step_optimizer": args.initial_first_step_optimizer,
        "lifted_first_step_backend": getattr(
            args, "lifted_first_step_backend", "same"
        ),
        "first_step_cg_seconds": args.first_step_cg_seconds,
        "fixed_point_solver": args.fixed_point_solver,
        "rank_tolerance": getattr(args, "rank_tolerance", 1e-12),
        "dense_verify_acceptance": getattr(
            args, "dense_verify_acceptance", False
        ),
        "lm_linear_solver": getattr(args, "lm_linear_solver", "normal"),
        "lm_tangent_slice": getattr(
            args, "lm_tangent_slice", "gauge-orthogonal"
        ),
        "lm_initial_damping": getattr(args, "lm_initial_damping", 1e-4),
        "lm_rdm_vectorization": getattr(
            args, "lm_rdm_vectorization", "full"
        ),
        "lm_jacobian_response_solver": getattr(
            args, "lm_jacobian_response_solver", "lstsq"
        ),
        "lm_jacobian_workers": getattr(args, "lm_jacobian_workers", 1),
        "target_source_fixed_point_solver": (
            args.target_source_fixed_point_solver
        ),
        "target_contraction": args.target_contraction,
        "time_predictor": args.time_predictor,
        "perturb_amplitudes": args.perturb_amplitudes,
        "perturbations_per_amplitude": args.perturbations_per_amplitude,
        "random_restarts": args.random_restarts,
        "random_seed": args.random_seed,
        "strict_retry": args.strict_retry,
        "fast_promote_below_bond_dimension": getattr(
            args, "fast_promote_below_bond_dimension", 0
        ),
        "optimizer_max_seconds": getattr(args, "optimizer_max_seconds", 600.0),
        "optimizer_iteration_cap": getattr(args, "optimizer_iteration_cap", 0),
        "first_step_lm_seconds": getattr(args, "first_step_lm_seconds", 600.0),
        "first_step_lm_iteration_cap": getattr(
            args, "first_step_lm_iteration_cap", 0
        ),
        "fast_promote_optimizer_seconds": getattr(
            args, "fast_promote_optimizer_seconds", 30.0
        ),
        "matrix_free_from_bond_dimension": getattr(
            args, "matrix_free_from_bond_dimension", 0
        ),
        "checkpoint_every": args.checkpoint_every,
        "optimizer": optimizer_policy(args),
    }
    if getattr(args, "runtime_promotion_factor", 0.0) > 0:
        policy["runtime_promotion"] = {
            "factor": args.runtime_promotion_factor,
            "window": args.runtime_promotion_window,
            "minimum_seconds": args.runtime_promotion_minimum_seconds,
        }
    return policy


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
    parser.add_argument("--initial-seed-A", type=Path, default=None)
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
    parser.add_argument("--initial-seed-lift-noise-amplitude", type=float, default=None)
    parser.add_argument("--embedding-noise-amplitude", type=float, default=1e-6)
    parser.add_argument("--embedding-seed", type=int, default=104_729)
    parser.add_argument(
        "--first-step-optimizer",
        choices=("cg-lm", "lm"),
        default="cg-lm",
        help="Optimizer used to fit each newly lifted bond-dimension handoff.",
    )
    parser.add_argument(
        "--initial-first-step-optimizer",
        choices=("cg-lm", "lm"),
        default=None,
        help=(
            "Optional optimizer override for the first segment only. This is "
            "useful when restarting from an already accepted same-D checkpoint."
        ),
    )
    parser.add_argument("--first-step-cg-seconds", type=float, default=900.0)
    parser.add_argument("--fixed-point-solver", choices=("dense", "fast"), default="dense")
    parser.add_argument("--rank-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--dense-verify-acceptance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recompute every candidate cost with the deterministic dense "
            "transfer fixed point before accepting it."
        ),
    )
    parser.add_argument(
        "--lm-linear-solver",
        choices=("normal", "svd"),
        default="normal",
    )
    parser.add_argument(
        "--lm-tangent-slice",
        choices=("gauge-orthogonal", "grassmann"),
        default="gauge-orthogonal",
    )
    parser.add_argument("--lm-initial-damping", type=float, default=1e-4)
    parser.add_argument(
        "--lm-rdm-vectorization",
        choices=("full", "hermitian"),
        default="full",
    )
    parser.add_argument(
        "--lm-jacobian-response-solver",
        choices=("lstsq", "dense-lu"),
        default="lstsq",
    )
    parser.add_argument("--lm-jacobian-workers", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=("dense-lm", "matrix-free-lm"),
        default="dense-lm",
    )
    parser.add_argument("--optimizer-max-seconds", type=float, default=600.0)
    parser.add_argument("--optimizer-iteration-cap", type=int, default=0)
    parser.add_argument("--first-step-lm-seconds", type=float, default=600.0)
    parser.add_argument("--first-step-lm-iteration-cap", type=int, default=0)
    parser.add_argument(
        "--matrix-free-from-bond-dimension",
        type=int,
        default=0,
        help=(
            "Use matrix-free LM at and above this D while retaining the "
            "requested --optimizer below it. Zero disables backend switching."
        ),
    )
    parser.add_argument(
        "--matrix-free-krylov-initial-iterations", type=int, default=64
    )
    parser.add_argument(
        "--matrix-free-krylov-solver",
        choices=("cg", "lsmr"),
        default="cg",
    )
    parser.add_argument(
        "--matrix-free-krylov-max-iterations", type=int, default=256
    )
    parser.add_argument(
        "--matrix-free-krylov-preconditioner",
        choices=(
            "none",
            "right-fixed-point",
            "right-fixed-point-stiefel",
        ),
        default="none",
    )
    parser.add_argument(
        "--matrix-free-krylov-relative-tolerance", type=float, default=0.1
    )
    parser.add_argument(
        "--matrix-free-krylov-minimum-relative-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--matrix-free-adaptive-krylov-tolerance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--matrix-free-recycle-krylov-solution",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--matrix-free-lsmr-condition-limit", type=float, default=1e12
    )
    parser.add_argument(
        "--matrix-free-fixed-point-response-solver",
        choices=("auto", "iterative", "dense-lu"),
        default="auto",
    )
    parser.add_argument(
        "--matrix-free-fixed-point-response-max-mb",
        type=float,
        default=128.0,
    )
    parser.add_argument("--matrix-free-adjoint-rtol", type=float, default=1e-8)
    parser.add_argument("--matrix-free-jvp-rtol", type=float, default=1e-8)
    parser.add_argument(
        "--matrix-free-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--matrix-free-dense-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--matrix-free-dense-rescue-seconds", type=float, default=3600.0
    )
    parser.add_argument(
        "--matrix-free-dense-rescue-iteration-cap", type=int, default=0
    )
    parser.add_argument(
        "--lifted-first-step-backend",
        choices=("same", "dense-lm"),
        default="same",
        help="Use dense LM for the first fixed target after every cross-D lift.",
    )
    parser.add_argument(
        "--target-source-fixed-point-solver",
        choices=("dense", "fast"),
        default="dense",
    )
    parser.add_argument("--target-contraction", choices=("tensor", "dense"), default="tensor")
    parser.add_argument(
        "--time-predictor", choices=("warm", "secant"), default="warm"
    )
    parser.add_argument("--perturb-amplitudes", default="0.1,0.3,0.6,1.0")
    parser.add_argument("--perturbations-per-amplitude", type=int, default=1)
    parser.add_argument("--random-restarts", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260702)
    parser.add_argument(
        "--strict-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fast-promote-below-bond-dimension",
        type=int,
        default=0,
        help=(
            "For D below this exclusive threshold, use one ordinary warm-start "
            "fit with no strict retry or distant seeds, then promote immediately "
            "if it misses the acceptance cost. Zero disables this policy."
        ),
    )
    parser.add_argument(
        "--fast-promote-optimizer-seconds",
        type=float,
        default=30.0,
        help="Per-solve recurrent LM cap below the fast-promotion threshold.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--maximum-step-seconds",
        type=float,
        default=0.0,
        help=(
            "Pause a child segment after checkpointing an accepted physical "
            "step that exceeds this wall-time limit. Zero disables the limit."
        ),
    )
    parser.add_argument("--runtime-promotion-factor", type=float, default=0.0)
    parser.add_argument("--runtime-promotion-window", type=int, default=8)
    parser.add_argument(
        "--runtime-promotion-minimum-seconds", type=float, default=60.0
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the first adaptive segment not yet recorded in manifest.json.",
    )
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
    initial_seed: Path | None = None,
    resume: bool = False,
) -> list[str]:
    fast_promote_threshold = getattr(
        args, "fast_promote_below_bond_dimension", 0
    )
    fast_promote = (
        fast_promote_threshold > 0
        and bond_dimension < fast_promote_threshold
    )
    effective_time_predictor = "warm" if fast_promote else args.time_predictor
    effective_perturbations = (
        0 if fast_promote else args.perturbations_per_amplitude
    )
    effective_random_restarts = 0 if fast_promote else args.random_restarts
    effective_strict_retry = False if fast_promote else args.strict_retry
    effective_optimizer_seconds = (
        getattr(args, "fast_promote_optimizer_seconds", 30.0)
        if fast_promote
        else getattr(args, "optimizer_max_seconds", 600.0)
    )
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
        (
            args.initial_first_step_optimizer
            if include_initial_key and args.initial_first_step_optimizer is not None
            else args.first_step_optimizer
        ),
        "--lifted-first-step-backend",
        getattr(args, "lifted_first_step_backend", "same"),
        "--first-step-cg-seconds",
        repr(args.first_step_cg_seconds),
        "--fixed-point-solver",
        args.fixed_point_solver,
        "--rank-tolerance",
        repr(getattr(args, "rank_tolerance", 1e-12)),
        (
            "--dense-verify-acceptance"
            if getattr(args, "dense_verify_acceptance", False)
            else "--no-dense-verify-acceptance"
        ),
        "--lm-linear-solver",
        getattr(args, "lm_linear_solver", "normal"),
        "--lm-tangent-slice",
        getattr(args, "lm_tangent_slice", "gauge-orthogonal"),
        "--lm-initial-damping",
        repr(getattr(args, "lm_initial_damping", 1e-4)),
        "--lm-rdm-vectorization",
        getattr(args, "lm_rdm_vectorization", "full"),
        "--lm-jacobian-response-solver",
        getattr(args, "lm_jacobian_response_solver", "lstsq"),
        "--lm-jacobian-workers",
        str(getattr(args, "lm_jacobian_workers", 1)),
        "--target-source-fixed-point-solver",
        args.target_source_fixed_point_solver,
        "--target-contraction",
        args.target_contraction,
        "--time-predictor",
        effective_time_predictor,
        "--perturb-amplitudes",
        args.perturb_amplitudes,
        "--perturbations-per-amplitude",
        str(effective_perturbations),
        "--random-restarts",
        str(effective_random_restarts),
        "--random-seed",
        str(args.random_seed),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--maximum-step-seconds",
        repr(getattr(args, "maximum_step_seconds", 0.0)),
        "--runtime-promotion-factor",
        repr(getattr(args, "runtime_promotion_factor", 0.0)),
        "--runtime-promotion-window",
        str(getattr(args, "runtime_promotion_window", 8)),
        "--runtime-promotion-minimum-seconds",
        repr(getattr(args, "runtime_promotion_minimum_seconds", 60.0)),
        "--strict-retry" if effective_strict_retry else "--no-strict-retry",
        "--optimizer-max-seconds",
        repr(effective_optimizer_seconds),
        "--optimizer-iteration-cap",
        str(getattr(args, "optimizer_iteration_cap", 0)),
        "--first-step-lm-seconds",
        repr(getattr(args, "first_step_lm_seconds", 600.0)),
        "--first-step-lm-iteration-cap",
        str(getattr(args, "first_step_lm_iteration_cap", 0)),
    ]
    if include_initial_key and args.initial_key is not None:
        command.extend(("--initial-key", args.initial_key))
    if initial_seed is not None:
        command.extend(
            (
                "--initial-seed-A",
                str(initial_seed),
                "--initial-seed-layout",
                "physical-left-right",
            )
        )
        initial_seed_lift_noise = getattr(
            args, "initial_seed_lift_noise_amplitude", None
        )
        if initial_seed_lift_noise is not None:
            command.extend(
                (
                    "--initial-seed-lift-noise-amplitude",
                    repr(initial_seed_lift_noise),
                )
            )
    if resume:
        command.append("--resume")
    optimizer = getattr(args, "optimizer", "dense-lm")
    matrix_free_from = getattr(args, "matrix_free_from_bond_dimension", 0)
    if matrix_free_from > 0 and bond_dimension >= matrix_free_from:
        optimizer = "matrix-free-lm"
    command.extend(("--optimizer", optimizer))
    if optimizer == "matrix-free-lm":
        command.extend(
            (
                "--matrix-free-krylov-initial-iterations",
                str(args.matrix_free_krylov_initial_iterations),
                "--matrix-free-krylov-solver",
                args.matrix_free_krylov_solver,
                "--matrix-free-krylov-max-iterations",
                str(args.matrix_free_krylov_max_iterations),
                "--matrix-free-krylov-preconditioner",
                args.matrix_free_krylov_preconditioner,
                "--matrix-free-krylov-relative-tolerance",
                repr(args.matrix_free_krylov_relative_tolerance),
                "--matrix-free-krylov-minimum-relative-tolerance",
                repr(args.matrix_free_krylov_minimum_relative_tolerance),
                "--matrix-free-adjoint-rtol",
                repr(args.matrix_free_adjoint_rtol),
                "--matrix-free-jvp-rtol",
                repr(args.matrix_free_jvp_rtol),
                "--matrix-free-lsmr-condition-limit",
                repr(args.matrix_free_lsmr_condition_limit),
                "--matrix-free-fixed-point-response-solver",
                args.matrix_free_fixed_point_response_solver,
                "--matrix-free-fixed-point-response-max-mb",
                repr(args.matrix_free_fixed_point_response_max_mb),
                (
                    "--matrix-free-adaptive-krylov-tolerance"
                    if args.matrix_free_adaptive_krylov_tolerance
                    else "--no-matrix-free-adaptive-krylov-tolerance"
                ),
                (
                    "--matrix-free-recycle-krylov-solution"
                    if args.matrix_free_recycle_krylov_solution
                    else "--no-matrix-free-recycle-krylov-solution"
                ),
                (
                    "--matrix-free-verbose"
                    if args.matrix_free_verbose
                    else "--no-matrix-free-verbose"
                ),
                (
                    "--matrix-free-dense-rescue"
                    if getattr(args, "matrix_free_dense_rescue", False)
                    else "--no-matrix-free-dense-rescue"
                ),
                "--matrix-free-dense-rescue-seconds",
                repr(getattr(args, "matrix_free_dense_rescue_seconds", 3600.0)),
                "--matrix-free-dense-rescue-iteration-cap",
                str(
                    getattr(
                        args,
                        "matrix_free_dense_rescue_iteration_cap",
                        0,
                    )
                ),
            )
        )
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


def resume_context(
    args: argparse.Namespace,
    ladder: tuple[int, ...],
    output: Path,
    manifest: dict[str, Any],
) -> tuple[Path, str, Path | None, int, float, int, bool]:
    expected = {
        "bond_dimensions": list(ladder),
        "block_length": args.block_length,
        "delta_t": args.delta_t,
        "requested_steps": args.steps,
        "base_time": args.base_time,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"adaptive resume setting differs for {key}")
    requested_optimizer_policy = optimizer_policy(args)
    if manifest.get("optimizer") != requested_optimizer_policy:
        raise ValueError("adaptive resume optimizer differs from manifest")
    if (
        "adaptive_policy" in manifest
        and manifest["adaptive_policy"] != adaptive_policy(args)
    ):
        raise ValueError("adaptive resume policy differs from manifest")
    if Path(manifest["initial_A"]).resolve() != args.initial_A.resolve():
        raise ValueError("adaptive resume initial tensor differs from manifest")
    if "initial_A_sha256" in manifest and manifest[
        "initial_A_sha256"
    ] != file_sha256(args.initial_A.resolve()):
        raise ValueError("adaptive resume initial tensor content differs")
    manifest_seed = manifest.get("initial_seed_A")
    requested_seed = (
        None if args.initial_seed_A is None else str(args.initial_seed_A.resolve())
    )
    if manifest_seed != requested_seed:
        raise ValueError("adaptive resume initial seed differs from manifest")
    if (
        args.initial_seed_A is not None
        and "initial_seed_A_sha256" in manifest
        and manifest["initial_seed_A_sha256"]
        != file_sha256(args.initial_seed_A.resolve())
    ):
        raise ValueError("adaptive resume initial seed content differs")
    if (
        "code_provenance" in manifest
        and manifest["code_provenance"] != code_provenance()
    ):
        raise ValueError("adaptive resume code or numerical runtime differs")

    source = args.initial_A.resolve()
    source_layout = args.initial_layout
    promotion_seed: Path | None = None
    accepted_steps = 0
    records = manifest.get("segments", [])
    if len(records) > len(ladder):
        raise ValueError("adaptive manifest has more segments than the requested ladder")
    for segment_index, record in enumerate(records):
        promotion_seed = None
        if record.get("index") != segment_index:
            raise ValueError("adaptive manifest segment indices are not contiguous")
        if record.get("bond_dimension") != ladder[segment_index]:
            raise ValueError("adaptive manifest bond dimension differs from ladder")
        segment_steps = int(record["completed_steps"])
        accepted_steps += segment_steps
        if segment_steps > 0:
            handoff = record.get("handoff_tensor")
            if handoff is None:
                raise ValueError("completed adaptive segment is missing its handoff")
            source = output / handoff
            if not source.exists():
                raise FileNotFoundError(f"adaptive handoff is missing: {source}")
            source_layout = "physical-left-right"
        promotion_seed_store = record.get("promotion_seed_tensor")
        if promotion_seed_store is not None:
            promotion_seed = output / promotion_seed_store
            if not promotion_seed.exists():
                raise FileNotFoundError(
                    f"adaptive promotion seed is missing: {promotion_seed}"
                )

    segment_index = len(records)
    if segment_index >= len(ladder):
        raise ValueError("adaptive run has no remaining bond dimension to resume")
    base_time = args.base_time + accepted_steps * args.delta_t
    segment_dir = output / f"segment_{segment_index:03d}_D{ladder[segment_index]}"
    resume_segment = segment_dir.exists()
    return (
        source,
        source_layout,
        promotion_seed,
        accepted_steps,
        base_time,
        segment_index,
        resume_segment,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ladder = parse_bond_dimensions(args.bond_dimensions)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    if args.delta_t <= 0:
        raise ValueError("--delta-t must be positive")
    if args.rank_tolerance <= 0:
        raise ValueError("--rank-tolerance must be positive")
    if args.lm_initial_damping <= 0:
        raise ValueError("--lm-initial-damping must be positive")
    if args.lm_jacobian_workers < 1:
        raise ValueError("--lm-jacobian-workers must be positive")
    if args.maximum_step_seconds < 0:
        raise ValueError("--maximum-step-seconds must be non-negative")
    if args.fast_promote_below_bond_dimension < 0:
        raise ValueError("fast-promotion bond threshold must be non-negative")
    if args.matrix_free_from_bond_dimension < 0:
        raise ValueError("matrix-free bond threshold must be non-negative")
    if args.runtime_promotion_factor < 0:
        raise ValueError("runtime-promotion factor must be non-negative")
    if args.runtime_promotion_window < 1:
        raise ValueError("runtime-promotion window must be positive")
    if args.runtime_promotion_minimum_seconds < 0:
        raise ValueError("runtime-promotion minimum must be non-negative")
    if args.optimizer_iteration_cap < 0 or args.first_step_lm_iteration_cap < 0:
        raise ValueError("LM iteration caps must be non-negative")
    if args.matrix_free_dense_rescue_iteration_cap < 0:
        raise ValueError("dense-rescue LM iteration cap must be non-negative")
    if args.matrix_free_dense_rescue and args.optimizer != "matrix-free-lm":
        raise ValueError("dense rescue requires --optimizer matrix-free-lm")
    if (
        args.optimizer_max_seconds <= 0
        or args.first_step_lm_seconds <= 0
        or args.fast_promote_optimizer_seconds <= 0
        or args.matrix_free_dense_rescue_seconds <= 0
    ):
        raise ValueError("optimizer wall-time caps must be positive")

    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError("adaptive --resume requires manifest.json")
        manifest = json.loads(manifest_path.read_text())
        (
            source,
            source_layout,
            promotion_seed,
            accepted_steps,
            base_time,
            first_segment_index,
            resume_segment,
        ) = resume_context(args, ladder, output, manifest)
        handoff_dir = output / "handoffs"
        if not handoff_dir.is_dir():
            raise FileNotFoundError("adaptive --resume requires the handoffs directory")
        manifest["status"] = "running"
        manifest["resumed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        manifest["accepted_steps"] = accepted_steps
        manifest["current_time"] = base_time
        manifest.pop("active_segment", None)
        manifest.pop("returncode", None)
        atomic_json(manifest_path, manifest)
    else:
        if output.exists():
            raise FileExistsError("adaptive output exists; choose a new directory")
        output.mkdir(parents=True)
        handoff_dir = output / "handoffs"
        handoff_dir.mkdir()
        source = args.initial_A.resolve()
        source_layout = args.initial_layout
        promotion_seed = None
        accepted_steps = 0
        base_time = float(args.base_time)
        first_segment_index = 0
        resume_segment = False
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "initial_A": str(source),
            "initial_A_sha256": file_sha256(source),
            "initial_seed_A": (
                None if args.initial_seed_A is None else str(args.initial_seed_A.resolve())
            ),
            "initial_seed_A_sha256": (
                None
                if args.initial_seed_A is None
                else file_sha256(args.initial_seed_A.resolve())
            ),
            "bond_dimensions": list(ladder),
            "block_length": args.block_length,
            "delta_t": args.delta_t,
            "requested_steps": args.steps,
            "base_time": args.base_time,
            "accepted_steps": 0,
            "current_time": base_time,
            "segments": [],
            "adaptive_policy": adaptive_policy(args),
            "code_provenance": code_provenance(),
        }
        requested_optimizer_policy = optimizer_policy(args)
        if requested_optimizer_policy is not None:
            manifest["optimizer"] = requested_optimizer_policy
        atomic_json(manifest_path, manifest)

    for segment_index in range(first_segment_index, len(ladder)):
        bond_dimension = ladder[segment_index]
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
            initial_seed=(
                args.initial_seed_A if segment_index == 0 else promotion_seed
            ),
            resume=(resume_segment and segment_index == first_segment_index),
        )
        print(
            f"adaptive segment={segment_index} D={bond_dimension} "
            f"base_time={base_time:.12g} remaining_steps={remaining}",
            flush=True,
        )
        manifest["active_segment"] = {
            "index": segment_index,
            "bond_dimension": bond_dimension,
            "directory": segment_dir.name,
            "base_time": base_time,
            "requested_steps": remaining,
            "resume": bool(resume_segment and segment_index == first_segment_index),
        }
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
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
        if status in RESUMABLE_STATUSES:
            manifest["status"] = status
            manifest["active_segment"].update(
                {
                    "completed_steps": segment_steps,
                    "current_time": base_time + segment_steps * args.delta_t,
                    "status": status,
                }
            )
            manifest["accepted_steps"] = accepted_steps + segment_steps
            manifest["current_time"] = base_time + segment_steps * args.delta_t
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            print(
                f"adaptive_status={status} "
                f"accepted={accepted_steps + segment_steps}/{args.steps}",
                flush=True,
            )
            return

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
        manifest.pop("active_segment", None)

        promotion_seed = None
        failure = metadata.get("failure")
        if status in PROMOTABLE_STATUSES and failure is not None:
            failed_tensor_store = failure.get("best_tensor_store")
            if failed_tensor_store is not None:
                failed_tensor_path = segment_dir / failed_tensor_store
                if not failed_tensor_path.exists():
                    raise FileNotFoundError(
                        "failed optimizer candidate declared in metadata is missing: "
                        f"{failed_tensor_path}"
                    )
                promotion_seed = failed_tensor_path
                segment_record["promotion_seed_tensor"] = str(
                    failed_tensor_path.relative_to(output)
                )

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
