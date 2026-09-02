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
from .embedding import (
    InitialLiftDiagnostics,
    lift_left_canonical_seed,
    product_circuit_left_canonical_seed,
)
from .fixed_target_cg import optimize_fixed_target_cg
from .horizontal_cg import (
    HorizontalCGOptions,
    HorizontalCGResult,
    optimize_fixed_target_horizontal_cg,
)
from .matrix_free_lm import (
    MatrixFreeLMOptions,
    MatrixFreeLMResult,
    optimize_fixed_target_matrix_free_lm,
)
from .optimizer import CGOptions, LMOptions, optimizer_right_fixed_point, optimize_tensor
from .predictor import trajectory_secant_predictor
from .protocol import (
    DEFAULT_PROTOCOL_PRESET,
    PROTOCOL_PRESETS,
    LocalEvolutionProtocol,
    NONINTEGRABLE_ISING,
    protocol_preset,
)
from .rdm import block_rdm
from .tensor_io import TensorLayout, load_tensor_file


OptimizerOptions = LMOptions | MatrixFreeLMOptions | HorizontalCGOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOL_PRESETS),
        default=DEFAULT_PROTOCOL_PRESET,
        help="Named Hamiltonian and Trotter convention preset.",
    )
    parser.add_argument("--initial-A", type=Path, required=True)
    parser.add_argument(
        "--initial-key",
        default=None,
        help="Array key when --initial-A is a keyed .npz archive.",
    )
    parser.add_argument(
        "--initial-layout",
        choices=("auto", "physical-left-right", "legacy-left-physical-right"),
        default="auto",
        help="Axis convention of --initial-A. Native LOMPS tensors are physical-first.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--base-time", type=float, required=True)
    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="Matched RDM length; defaults to the selected protocol preset.",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=None,
        help="Trotter step size; defaults to the selected protocol preset.",
    )
    parser.add_argument(
        "--trotter-order",
        type=int,
        choices=(2,),
        default=None,
        help="Trotter order; LOMPS currently supports only second order.",
    )
    parser.add_argument(
        "--g",
        type=float,
        default=None,
        help="Override the preset coefficient of X tensor I in the bond Hamiltonian.",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=None,
        help="Override the preset coefficient of Z tensor I in the bond Hamiltonian.",
    )
    parser.add_argument(
        "--J",
        "--coupling-J",
        dest="J",
        type=float,
        default=None,
        help="Override the preset coefficient of Z tensor Z in the bond Hamiltonian.",
    )
    parser.add_argument(
        "--symmetric-transverse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Split the transverse field equally over both sites of each bond gate. "
            "The historical integrable benchmark uses --no-symmetric-transverse."
        ),
    )
    parser.add_argument(
        "--bond-dimension",
        type=int,
        default=0,
        help=(
            "Trajectory bond dimension. Defaults to the input tensor bond "
            "dimension; if larger, the input is used as the exact first-target "
            "source and a deterministic lifted seed starts the optimizer."
        ),
    )
    parser.add_argument(
        "--initial-seed-A",
        type=Path,
        default=None,
        help=(
            "Optional optimizer seed tensor. The physical first target is still "
            "built from --initial-A; this tensor only initializes the trajectory "
            "manifold and is lifted if its bond dimension is smaller."
        ),
    )
    parser.add_argument(
        "--initial-seed-key",
        default=None,
        help="Array key when --initial-seed-A is a keyed .npz archive.",
    )
    parser.add_argument(
        "--initial-seed-layout",
        choices=("auto", "physical-left-right", "legacy-left-physical-right"),
        default="auto",
        help="Axis convention of --initial-seed-A.",
    )
    parser.add_argument(
        "--initial-seed-lift-noise-amplitude",
        type=float,
        default=None,
        help=(
            "Noise for lifting --initial-seed-A to --bond-dimension. If omitted, "
            "uses 1e-4 for trajectory D>=20 and --embedding-noise-amplitude otherwise."
        ),
    )
    parser.add_argument(
        "--odd-parity-warning-threshold",
        type=float,
        default=NONINTEGRABLE_ISING.odd_parity_warning_threshold,
    )
    parser.add_argument(
        "--target-contraction",
        choices=("tensor", "dense"),
        default=NONINTEGRABLE_ISING.target_contraction,
        help=(
            "How to build evolved finite-window targets. 'tensor' applies "
            "local gates to density-tensor axes without materializing the full "
            "brickwall unitary; 'dense' keeps the old dense-unitary path."
        ),
    )
    parser.add_argument(
        "--target-source-fixed-point-solver",
        choices=("dense", "fast"),
        default=NONINTEGRABLE_ISING.target_source_fixed_point_solver,
        help=(
            "Fixed-point solver used only when building the evolved light-cone "
            "target. This can be set independently from --fixed-point-solver."
        ),
    )
    parser.add_argument(
        "--accept-cost",
        type=float,
        default=1e-14,
        help=(
            "Acceptance threshold for recurrent trajectory steps. The default "
            "is relaxed relative to the first product fit so continuation "
            "does not spend time chasing the floating-point floor."
        ),
    )
    parser.add_argument(
        "--first-step-accept-cost",
        type=float,
        default=3e-16,
        help=(
            "Acceptance threshold only for the first low-D source to "
            "trajectory-D fit. Later trajectory steps use --accept-cost."
        ),
    )
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
    parser.add_argument(
        "--dense-verify-acceptance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recompute the final candidate cost with the deterministic dense "
            "transfer fixed point before accepting it."
        ),
    )
    parser.add_argument(
        "--lm-linear-solver",
        choices=("normal", "svd"),
        default="normal",
        help=(
            "Linear solver inside LM optimizer evaluations. 'normal' builds "
            "the dense Jacobian but solves damped normal equations by Cholesky; "
            "'svd' keeps the older SVD-LM step and rank diagnostics."
        ),
    )
    parser.add_argument(
        "--lm-tangent-slice",
        choices=("gauge-orthogonal", "grassmann"),
        default="gauge-orthogonal",
        help=(
            "Quotient-tangent representative used by dense LM. "
            "'gauge-orthogonal' is the established true-gauge basis; "
            "'grassmann' uses the cheaper TDVP left-gauge slice."
        ),
    )
    parser.add_argument(
        "--lm-initial-damping",
        type=float,
        default=1e-4,
        help="Initial scalar damping for each dense LM fixed-target solve.",
    )
    parser.add_argument(
        "--lm-rdm-vectorization",
        choices=("full", "hermitian"),
        default="full",
        help=(
            "Real coordinate representation of the Hermitian RDM residual "
            "and dense Jacobian. 'hermitian' preserves Frobenius geometry "
            "while avoiding duplicated matrix entries."
        ),
    )
    parser.add_argument(
        "--lm-jacobian-response-solver",
        choices=("lstsq", "dense-lu"),
        default="lstsq",
        help=(
            "Fixed-point response solve used while constructing the dense "
            "LM Jacobian. 'dense-lu' factors the stabilized square response "
            "operator once per evaluation; 'lstsq' is the legacy path."
        ),
    )
    parser.add_argument(
        "--lm-jacobian-workers",
        type=int,
        default=1,
        help=(
            "Number of deterministic independent tangent-batch workers used "
            "to construct the dense LM Jacobian."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("dense-lm", "matrix-free-lm", "horizontal-cg"),
        default="dense-lm",
        help=(
            "Fixed-target optimizer backend. 'dense-lm' is the established "
            "explicit-J solver. 'matrix-free-lm' uses horizontal LM with "
            "inner CG. 'horizontal-cg' uses gauge-horizontal nonlinear CG "
            "with an analytic VJP. Neither alternative constructs the RDM "
            "Jacobian or tangent basis."
        ),
    )
    parser.add_argument(
        "--optimizer-max-seconds",
        type=float,
        default=600.0,
        help="Wall-time cap for each recurrent LM solve.",
    )
    parser.add_argument(
        "--optimizer-iteration-cap",
        type=int,
        default=0,
        help=(
            "Maximum number of recorded outer LM iterations per recurrent "
            "solve. Zero retains the legacy iteration limit."
        ),
    )
    parser.add_argument(
        "--first-step-lm-seconds",
        type=float,
        default=600.0,
        help=(
            "Wall-time cap for LM polishing of a lifted first step; the "
            "preceding CG stage has its own --first-step-cg-seconds cap."
        ),
    )
    parser.add_argument(
        "--first-step-lm-iteration-cap",
        type=int,
        default=0,
        help=(
            "Maximum recorded outer LM iterations in first-step polishing. "
            "Zero retains the legacy iteration limit."
        ),
    )
    parser.add_argument(
        "--matrix-free-krylov-initial-iterations",
        type=int,
        default=64,
        help="Initial Krylov iteration cap for matrix-free LM.",
    )
    parser.add_argument(
        "--matrix-free-krylov-solver",
        choices=("cg", "lsmr"),
        default="cg",
        help=(
            "Matrix-free linear solver. 'cg' acts on damped normal equations; "
            "'lsmr' acts on the rectangular damped least-squares operator."
        ),
    )
    parser.add_argument(
        "--matrix-free-krylov-max-iterations",
        type=int,
        default=256,
        help="Largest adaptively selected inner-CG iteration cap.",
    )
    parser.add_argument(
        "--matrix-free-krylov-preconditioner",
        choices=(
            "none",
            "right-fixed-point",
            "right-fixed-point-stiefel",
        ),
        default="none",
        help=(
            "Inner-CG preconditioner. The Stiefel variant avoids structured "
            "gauge solves inside CG and exactly gauge-projects the completed "
            "LM direction."
        ),
    )
    parser.add_argument(
        "--matrix-free-krylov-relative-tolerance",
        type=float,
        default=0.1,
        help="Initial relative residual tolerance for the Krylov solve.",
    )
    parser.add_argument(
        "--matrix-free-krylov-minimum-relative-tolerance",
        type=float,
        default=1e-6,
        help="Smallest Krylov tolerance used near convergence.",
    )
    parser.add_argument(
        "--matrix-free-adaptive-krylov-tolerance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tighten the Krylov tolerance as the outer residual falls.",
    )
    parser.add_argument(
        "--matrix-free-recycle-krylov-solution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Warm-start each Krylov solve from the preceding accepted step.",
    )
    parser.add_argument(
        "--matrix-free-lsmr-condition-limit",
        type=float,
        default=1e12,
        help="Condition-estimate stopping limit used only by LSMR.",
    )
    parser.add_argument(
        "--matrix-free-fixed-point-response-solver",
        choices=("auto", "iterative", "dense-lu"),
        default="auto",
        help=(
            "Solver reused by JVP/VJP fixed-point responses. 'auto' uses a "
            "cached dense LU only when its estimated storage fits the limit."
        ),
    )
    parser.add_argument(
        "--matrix-free-fixed-point-response-max-mb",
        type=float,
        default=128.0,
        help="Maximum cached response-factor storage selected by 'auto'.",
    )
    parser.add_argument(
        "--matrix-free-adjoint-rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for matrix-free adjoint fixed-point solves.",
    )
    parser.add_argument(
        "--matrix-free-jvp-rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for fixed-point derivatives inside JVPs.",
    )
    parser.add_argument(
        "--matrix-free-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print one diagnostic line per outer matrix-free LM iteration.",
    )
    parser.add_argument(
        "--matrix-free-dense-rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If matrix-free LM misses the acceptance cost, continue the same "
            "fixed target once with dense LM before declaring failure."
        ),
    )
    parser.add_argument(
        "--matrix-free-dense-rescue-seconds",
        type=float,
        default=3600.0,
        help="Wall-time cap for the same-target dense-LM rescue.",
    )
    parser.add_argument(
        "--matrix-free-dense-rescue-iteration-cap",
        type=int,
        default=0,
        help="Dense-rescue LM iteration cap; zero keeps the legacy limit.",
    )
    parser.add_argument(
        "--lifted-first-step-backend",
        choices=("same", "dense-lm"),
        default="same",
        help=(
            "Backend for the first target after a cross-D lift. 'same' uses "
            "the recurrent backend; 'dense-lm' reserves explicit LM for the "
            "handoff fit."
        ),
    )
    parser.add_argument(
        "--horizontal-cg-preconditioner",
        choices=(
            "none",
            "right-fixed-point",
            "grassmann-slice",
            "grassmann-right-fixed-point",
        ),
        default="grassmann-right-fixed-point",
        help="Preconditioner for recurrent gauge-horizontal nonlinear CG.",
    )
    parser.add_argument(
        "--horizontal-cg-conjugacy-metric",
        choices=("horizontal", "historical-slice"),
        default="historical-slice",
        help="Metric representation used for nonlinear-CG conjugacy.",
    )
    parser.add_argument(
        "--horizontal-cg-line-interpolation",
        choices=("bisection", "secant"),
        default="secant",
        help="Interpolation rule inside horizontal-CG line searches.",
    )
    parser.add_argument("--horizontal-cg-restart", type=int, default=100)
    parser.add_argument(
        "--horizontal-cg-maximum-line-search-evaluations",
        type=int,
        default=20,
    )
    parser.add_argument("--horizontal-cg-adjoint-rtol", type=float, default=1e-8)
    parser.add_argument(
        "--horizontal-cg-gauge-projector",
        choices=("dense", "structured"),
        default="dense",
    )
    parser.add_argument(
        "--horizontal-cg-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--time-predictor",
        choices=("warm", "secant"),
        default="warm",
        help=(
            "Optimizer seed for recurrent steps. 'secant' extrapolates the two "
            "latest gauge-continuous tensors, retracts to left-canonical form, "
            "and falls back to the warm tensor whenever its seed cost is worse."
        ),
    )
    parser.add_argument("--perturb-amplitudes", type=str, default="0.3,0.6,1,2,4,8")
    parser.add_argument("--perturbations-per-amplitude", type=int, default=2)
    parser.add_argument("--random-restarts", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260702)
    parser.add_argument(
        "--strict-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retry each failed LM seed with stricter stopping controls. "
            "Use --no-strict-retry for independent restart batches where each "
            "seed receives only one primary LM solve."
        ),
    )
    parser.add_argument(
        "--embedding-noise-amplitude",
        type=float,
        default=1e-6,
        help=(
            "Noise used only to form the lifted optimizer seed. The first "
            "RDM target is still built from the original input source."
        ),
    )
    parser.add_argument("--embedding-seed", type=int, default=104_729)
    parser.add_argument(
        "--initial-seed-mode",
        choices=("auto", "embedding", "circuit"),
        default="embedding",
        help=(
            "How to choose the optimizer seed for low-D starts. The default "
            "'embedding' uses the historical noisy product embedding only as "
            "a start point. 'auto' uses the product-circuit seed when possible "
            "and otherwise falls back to embedding."
        ),
    )
    parser.add_argument(
        "--circuit-lift-mixing-amplitude",
        type=float,
        default=1e-3,
        help=(
            "Stiefel mixing amplitude for product-circuit initial seeds; "
            "positive values break the exact period-two transfer degeneracy."
        ),
    )
    parser.add_argument(
        "--circuit-lift-seed",
        type=int,
        default=2,
        help="Random seed for product-circuit Stiefel mixing.",
    )
    parser.add_argument(
        "--embedding-candidate-seeds",
        type=str,
        default="",
        help=(
            "Optional comma-separated lift seeds. When provided for a low-D "
            "initial state, LOMPS screens these seeds against the exact first "
            "target and starts from the seed with the lowest screening cost."
        ),
    )
    parser.add_argument(
        "--embedding-screen-max-iterations",
        type=int,
        default=25,
        help="Maximum LM iterations per initial-lift screening candidate.",
    )
    parser.add_argument(
        "--embedding-screen-seconds",
        type=float,
        default=30.0,
        help="Maximum wall seconds per initial-lift screening candidate.",
    )
    parser.add_argument(
        "--embedding-screen-fixed-point-solver",
        choices=("same", "dense", "fast"),
        default="same",
        help="Fixed-point solver used for initial-lift screening.",
    )
    parser.add_argument(
        "--first-step-optimizer",
        choices=("cg-lm", "lm"),
        default="cg-lm",
        help=(
            "Optimizer used for the first low-D-to-high-D target. The default "
            "'cg-lm' runs analytic fixed-target Grassmann CG and then polishes "
            "the same target with LM."
        ),
    )
    parser.add_argument("--first-step-cg-max-iterations", type=int, default=40_000)
    parser.add_argument("--first-step-cg-seconds", type=float, default=900.0)
    parser.add_argument("--first-step-cg-gradient-tolerance", type=float, default=1e-11)
    parser.add_argument(
        "--first-step-cg-initial-step",
        type=float,
        default=0.0,
        help="Initial CG line-search step; non-positive means choose from the gradient norm.",
    )
    parser.add_argument(
        "--first-step-cg-restart",
        type=int,
        default=100,
        help="Restart period for first-step nonlinear CG.",
    )
    parser.add_argument(
        "--first-step-cg-precondition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the right-fixed-point preconditioner in first-step CG.",
    )
    parser.add_argument(
        "--first-step-cg-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print first-step CG iteration diagnostics.",
    )
    parser.add_argument(
        "--first-step-cg-fixed-point-solver",
        choices=("same", "dense", "fast"),
        default="same",
        help="Fixed-point solver used by first-step CG.",
    )
    parser.add_argument("--initial-canonical-tolerance", type=float, default=1e-10)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--run-time-limit", type=float, default=0.0)
    parser.add_argument(
        "--maximum-step-seconds",
        type=float,
        default=0.0,
        help=(
            "Pause after checkpointing an accepted physical step whose total "
            "wall time exceeds this limit. Zero disables the limit."
        ),
    )
    parser.add_argument(
        "--runtime-promotion-factor",
        type=float,
        default=0.0,
        help=(
            "After an accepted step, request bond-dimension promotion when "
            "its wall time exceeds this factor times the rolling median. "
            "Zero disables runtime-triggered promotion."
        ),
    )
    parser.add_argument("--runtime-promotion-window", type=int, default=8)
    parser.add_argument(
        "--runtime-promotion-minimum-seconds", type=float, default=60.0
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help=(
            "Pause after this total accepted-step index. This is an execution "
            "control, so it may be changed between resumes without changing "
            "the stored trajectory policy; zero disables the limit."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def step_limit_reached(completed: int, stop_after_step: int) -> bool:
    """Return whether an execution-only accepted-step limit has been reached."""
    return stop_after_step > 0 and completed >= stop_after_step


def step_runtime_limit_exceeded(step_seconds: float, maximum_seconds: float) -> bool:
    """Return whether an accepted physical step exceeded its execution limit."""

    return maximum_seconds > 0 and step_seconds > maximum_seconds


def runtime_promotion_diagnostic(
    step_seconds: float,
    preceding_step_seconds: list[float],
    *,
    factor: float,
    window: int,
    minimum_seconds: float,
) -> dict[str, float | bool] | None:
    """Return the rolling-median runtime trigger diagnostic, if enabled."""

    if factor <= 0 or len(preceding_step_seconds) < window:
        return None
    reference = float(np.median(preceding_step_seconds[-window:]))
    ratio = float(step_seconds / reference) if reference > 0 else float("inf")
    return {
        "reference_median_seconds": reference,
        "ratio": ratio,
        "triggered": bool(
            step_seconds >= minimum_seconds and ratio >= factor
        ),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, value)
    temporary.replace(path)


def fixed_point_info_json(info: dict[str, float | complex]) -> dict[str, float]:
    """Return JSON-safe fixed-point diagnostics."""

    result: dict[str, float] = {}
    for key, value in info.items():
        if isinstance(value, (complex, np.complexfloating)):
            complex_value = complex(value)
            result[f"{key}_real"] = float(complex_value.real)
            result[f"{key}_imag"] = float(complex_value.imag)
        else:
            result[key] = float(value)
    return result


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def append_matrix_free_history(
    path: Path,
    *,
    step: int,
    attempt: str,
    result: object,
) -> None:
    """Persist every inner-CG diagnostic for one matrix-free fit."""

    if not isinstance(result, MatrixFreeLMResult):
        return
    for record in result.history:
        append_csv(
            path,
            {
                "step": step,
                "attempt": attempt,
                "krylov_solver": result.krylov_solver,
                "fixed_point_response_solver": (
                    result.fixed_point_response_solver
                ),
                "fixed_point_response_storage_bytes": (
                    result.fixed_point_response_storage_bytes
                ),
                **asdict(record),
            },
        )


def append_horizontal_cg_history(
    path: Path,
    *,
    step: int,
    attempt: str,
    result: object,
) -> None:
    """Persist nonlinear-CG line-search diagnostics for one fixed-target fit."""

    if not isinstance(result, HorizontalCGResult):
        return
    for record in result.history:
        append_csv(
            path,
            {
                "step": step,
                "attempt": attempt,
                **asdict(record),
            },
        )


def append_optimizer_history(
    matrix_free_path: Path,
    horizontal_cg_path: Path,
    *,
    step: int,
    attempt: str,
    result: object,
) -> None:
    """Persist backend-specific optimizer diagnostics when available."""

    append_matrix_free_history(
        matrix_free_path,
        step=step,
        attempt=attempt,
        result=result,
    )
    append_horizontal_cg_history(
        horizontal_cg_path,
        step=step,
        attempt=attempt,
        result=result,
    )


def optimizer_evaluation_count(result: object) -> int:
    """Return objective evaluations, falling back to outer history length."""

    if hasattr(result, "objective_evaluations"):
        return int(getattr(result, "objective_evaluations"))
    return len(getattr(result, "history", ()))


def load_initial(
    path: Path,
    *,
    key: str | None = None,
    layout: TensorLayout = "auto",
) -> np.ndarray:
    tensor, _ = load_tensor_file(path, key=key, layout=layout)
    return tensor


def parse_seed_list(text: str) -> tuple[int, ...]:
    """Return a de-duplicated integer seed tuple from comma-separated text."""

    seeds: list[int] = []
    seen: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return tuple(seeds)


def effective_initial_seed_lift_noise(
    *,
    requested: float | None,
    trajectory_bond_dimension: int,
    embedding_noise_amplitude: float,
) -> float:
    """Return the external-seed lift noise after applying LOMPS defaults."""

    if requested is not None:
        return float(requested)
    if trajectory_bond_dimension >= 20:
        return 1e-4
    return float(embedding_noise_amplitude)


def effective_first_step_accept_cost(
    requested: float | None,
    default_accept_cost: float,
) -> float:
    """Return the first-step acceptance threshold after applying defaults."""

    return float(default_accept_cost if requested is None else requested)


def options(
    accept_cost: float,
    rank_tolerance: float,
    fixed_point_solver: str = "dense",
    linear_solver: str = "normal",
    maximum_seconds: float = 600.0,
    iteration_cap: int = 0,
    tangent_slice: str = "gauge_orthogonal",
    initial_damping: float = 1e-4,
    rdm_vectorization: str = "full",
    jacobian_response_solver: str = "lstsq",
    jacobian_workers: int = 1,
) -> tuple[LMOptions, LMOptions]:
    max_iterations = 40_000 if iteration_cap <= 0 else iteration_cap - 1
    primary = LMOptions(
        max_iterations=max_iterations,
        gradient_tolerance=1e-11,
        cost_tolerance=accept_cost,
        rank_tolerance=rank_tolerance,
        tangent_slice=tangent_slice,
        initial_damping=initial_damping,
        fixed_point_solver=fixed_point_solver,
        linear_solver=linear_solver,
        rdm_vectorization=rdm_vectorization,
        jacobian_response_solver=jacobian_response_solver,
        jacobian_workers=jacobian_workers,
        plateau_window=50,
        plateau_relative_cost_drop=1e-4,
        plateau_absolute_cost_drop=1e-20,
        maximum_seconds=maximum_seconds,
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


def matrix_free_options(
    accept_cost: float,
    rank_tolerance: float,
    fixed_point_solver: str,
    *,
    krylov_initial_iterations: int,
    krylov_max_iterations: int,
    krylov_relative_tolerance: float,
    krylov_minimum_relative_tolerance: float,
    adaptive_krylov_tolerance: bool,
    adjoint_rtol: float,
    jvp_fixed_point_rtol: float,
    krylov_preconditioner: str = "none",
    verbose: bool = False,
    krylov_solver: str = "cg",
    recycle_krylov_solution: bool = False,
    lsmr_condition_limit: float = 1e12,
    fixed_point_response_solver: str = "auto",
    fixed_point_response_maximum_bytes: int = 128 * 1024**2,
    maximum_seconds: float = 600.0,
    iteration_cap: int = 0,
) -> tuple[MatrixFreeLMOptions, MatrixFreeLMOptions]:
    """Production and strict controls for basis-free horizontal LM."""

    max_iterations = 40_000 if iteration_cap <= 0 else iteration_cap - 1
    primary = MatrixFreeLMOptions(
        max_iterations=max_iterations,
        gradient_tolerance=1e-11,
        cost_tolerance=accept_cost,
        gauge_tolerance=max(rank_tolerance, 1e-10),
        fixed_point_solver=fixed_point_solver,
        fixed_point_response_solver=fixed_point_response_solver,
        fixed_point_response_maximum_bytes=(
            fixed_point_response_maximum_bytes
        ),
        krylov_solver=krylov_solver,
        krylov_initial_iterations=krylov_initial_iterations,
        krylov_max_iterations=krylov_max_iterations,
        krylov_relative_tolerance=krylov_relative_tolerance,
        krylov_minimum_relative_tolerance=(
            krylov_minimum_relative_tolerance
        ),
        adaptive_krylov_tolerance=adaptive_krylov_tolerance,
        krylov_preconditioner=krylov_preconditioner,
        recycle_krylov_solution=recycle_krylov_solution,
        lsmr_condition_limit=lsmr_condition_limit,
        adjoint_rtol=adjoint_rtol,
        jvp_fixed_point_rtol=jvp_fixed_point_rtol,
        reduce_damping_only_on_krylov_convergence=False,
        plateau_window=50,
        plateau_relative_cost_drop=1e-4,
        plateau_absolute_cost_drop=1e-20,
        maximum_seconds=maximum_seconds,
        verbose=verbose,
    )
    strict = replace(
        primary,
        gradient_tolerance=1e-13,
        krylov_minimum_relative_tolerance=min(
            primary.krylov_minimum_relative_tolerance, 1e-8
        ),
        adjoint_rtol=min(primary.adjoint_rtol, 1e-10),
        jvp_fixed_point_rtol=min(primary.jvp_fixed_point_rtol, 1e-10),
        plateau_window=200,
        plateau_relative_cost_drop=1e-9,
        plateau_absolute_cost_drop=1e-24,
    )
    return primary, strict


def horizontal_cg_options(
    accept_cost: float,
    fixed_point_solver: str,
    *,
    preconditioner: str = "grassmann_right_fixed_point",
    conjugacy_metric: str = "historical_slice",
    line_search_interpolation: str = "secant",
    restart: int = 100,
    maximum_line_search_evaluations: int = 20,
    adjoint_rtol: float = 1e-8,
    gauge_projector: str = "dense",
    maximum_seconds: float = 600.0,
    iteration_cap: int = 0,
    verbose: bool = False,
) -> tuple[HorizontalCGOptions, HorizontalCGOptions]:
    """Build recurrent and strict gauge-horizontal nonlinear-CG controls."""

    max_iterations = 40_000 if iteration_cap <= 0 else iteration_cap
    primary = HorizontalCGOptions(
        max_iterations=max_iterations,
        cost_tolerance=accept_cost,
        gradient_tolerance=1e-11,
        preconditioner=preconditioner,
        conjugacy_metric=conjugacy_metric,
        line_search_interpolation=line_search_interpolation,
        restart=restart,
        maximum_line_search_evaluations=maximum_line_search_evaluations,
        adjoint_rtol=adjoint_rtol,
        gauge_projector=gauge_projector,
        maximum_seconds=maximum_seconds,
        fixed_point_solver=fixed_point_solver,
        verbose=verbose,
    )
    strict = replace(
        primary,
        gradient_tolerance=1e-13,
        maximum_line_search_evaluations=max(
            40, primary.maximum_line_search_evaluations
        ),
    )
    return primary, strict


def run_fixed_target_optimizer(
    seed: np.ndarray,
    target: np.ndarray,
    block_length: int,
    optimizer_options: OptimizerOptions,
    *,
    initial_fixed_point: np.ndarray | None = None,
) -> tuple[np.ndarray, object]:
    """Dispatch one fixed-target fit without changing its target."""

    if isinstance(optimizer_options, MatrixFreeLMOptions):
        return optimize_fixed_target_matrix_free_lm(
            seed,
            target,
            block_length,
            optimizer_options,
            initial_fixed_point=initial_fixed_point,
        )
    if isinstance(optimizer_options, HorizontalCGOptions):
        return optimize_fixed_target_horizontal_cg(
            seed,
            target,
            block_length,
            optimizer_options,
        )
    return optimize_tensor(
        seed,
        target,
        block_length,
        optimizer_options,
        initial_fixed_point=initial_fixed_point,
    )


def first_step_cg_options(
    args: argparse.Namespace,
    *,
    accept_cost: float,
    rank_tolerance: float,
    default_fixed_point_solver: str,
) -> CGOptions:
    """Build first-step CG controls from CLI arguments."""

    fixed_point_solver = (
        default_fixed_point_solver
        if args.first_step_cg_fixed_point_solver == "same"
        else args.first_step_cg_fixed_point_solver
    )
    return CGOptions(
        max_iterations=args.first_step_cg_max_iterations,
        gradient_tolerance=args.first_step_cg_gradient_tolerance,
        cost_tolerance=accept_cost,
        rank_tolerance=rank_tolerance,
        initial_step=(
            None
            if args.first_step_cg_initial_step <= 0
            else args.first_step_cg_initial_step
        ),
        precondition=args.first_step_cg_precondition,
        restart=args.first_step_cg_restart,
        maximum_seconds=args.first_step_cg_seconds,
        fixed_point_solver=fixed_point_solver,
        verbose=args.first_step_cg_verbose,
    )


def protocol_from_args(args: argparse.Namespace):
    preset_name = getattr(args, "protocol", DEFAULT_PROTOCOL_PRESET)
    base = protocol_preset(preset_name)
    block_length = (
        base.block_length if getattr(args, "block_length", None) is None else args.block_length
    )
    delta_t = base.delta_t if getattr(args, "delta_t", None) is None else args.delta_t
    if block_length < 1:
        raise ValueError("--block-length must be positive")
    if delta_t <= 0:
        raise ValueError("--delta-t must be positive")
    if args.odd_parity_warning_threshold < 0:
        raise ValueError("--odd-parity-warning-threshold must be non-negative")
    name_prefix = base.name.rsplit("_L", maxsplit=1)[0]
    protocol = replace(
        base,
        name=f"{name_prefix}_L{block_length}",
        block_length=block_length,
        delta_t=delta_t,
        trotter_order=(
            base.trotter_order
            if getattr(args, "trotter_order", None) is None
            else args.trotter_order
        ),
        g=base.g if getattr(args, "g", None) is None else args.g,
        h=base.h if getattr(args, "h", None) is None else args.h,
        J=base.J if getattr(args, "J", None) is None else args.J,
        symmetric_transverse=(
            base.symmetric_transverse
            if getattr(args, "symmetric_transverse", None) is None
            else args.symmetric_transverse
        ),
        odd_parity_warning_threshold=args.odd_parity_warning_threshold,
        target_contraction=args.target_contraction,
        target_source_fixed_point_solver=args.target_source_fixed_point_solver,
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


def select_initial_seed(
    initial_source: np.ndarray,
    trajectory_bond_dimension: int,
    *,
    protocol: LocalEvolutionProtocol,
    primary: OptimizerOptions,
    initial_seed_mode: str,
    embedding_noise_amplitude: float,
    embedding_seed: int,
    embedding_candidate_seeds: tuple[int, ...],
    embedding_screen_max_iterations: int,
    embedding_screen_seconds: float,
    embedding_screen_fixed_point_solver: str,
    circuit_lift_mixing_amplitude: float,
    circuit_lift_seed: int,
    canonical_tolerance: float,
) -> tuple[np.ndarray, InitialLiftDiagnostics, list[dict[str, Any]]]:
    """Choose the initial high-D seed, optionally screening lift seeds.

    Screening is only meaningful when a lower-bond source is lifted into a
    larger optimizer manifold. The exact first target is still computed from
    ``initial_source``; candidate seeds are ranked by a bounded fit to that
    unchanged first-step target.
    """

    use_circuit = initial_seed_mode in ("auto", "circuit")
    if use_circuit and initial_source.shape[1] < trajectory_bond_dimension:
        try:
            seed_tensor, diagnostics = product_circuit_left_canonical_seed(
                initial_source,
                trajectory_bond_dimension,
                protocol,
                mixing_amplitude=circuit_lift_mixing_amplitude,
                seed=circuit_lift_seed,
                canonical_tolerance=canonical_tolerance,
            )
            return seed_tensor, diagnostics, []
        except ValueError:
            if initial_seed_mode == "circuit":
                raise

    if initial_seed_mode == "circuit":
        raise ValueError("circuit seed mode requires a lower-D product input")

    if not embedding_candidate_seeds or initial_source.shape[1] == trajectory_bond_dimension:
        seed_tensor, diagnostics = lift_left_canonical_seed(
            initial_source,
            trajectory_bond_dimension,
            noise_amplitude=embedding_noise_amplitude,
            seed=embedding_seed,
            diagnostic_block_length=protocol.block_length,
            canonical_tolerance=canonical_tolerance,
        )
        return seed_tensor, diagnostics, []

    if embedding_screen_max_iterations < 0:
        raise ValueError("--embedding-screen-max-iterations must be non-negative")
    if embedding_screen_seconds < 0:
        raise ValueError("--embedding-screen-seconds must be non-negative")

    screen_solver = (
        primary.fixed_point_solver
        if embedding_screen_fixed_point_solver == "same"
        else embedding_screen_fixed_point_solver
    )
    screen_options = replace(
        primary,
        max_iterations=embedding_screen_max_iterations,
        maximum_seconds=embedding_screen_seconds,
        cost_tolerance=0.0,
        fixed_point_solver=screen_solver,
        verbose=False,
    )
    target = protocol.target_rdm(initial_source)
    best: tuple[float, np.ndarray, InitialLiftDiagnostics] | None = None
    rows: list[dict[str, Any]] = []
    for index, candidate_seed in enumerate(embedding_candidate_seeds):
        seed_tensor, diagnostics = lift_left_canonical_seed(
            initial_source,
            trajectory_bond_dimension,
            noise_amplitude=embedding_noise_amplitude,
            seed=candidate_seed,
            diagnostic_block_length=protocol.block_length,
            canonical_tolerance=canonical_tolerance,
        )
        started = time.perf_counter()
        _, result = run_fixed_target_optimizer(
            seed_tensor,
            target,
            protocol.block_length,
            screen_options,
        )
        seconds = time.perf_counter() - started
        row = {
            "candidate_index": index,
            "seed": int(candidate_seed),
            "noise_amplitude": float(diagnostics.noise_amplitude),
            "cost": float(result.cost),
            "target_residual": float(result.residual_norm),
            "status": result.status,
            "evaluations": optimizer_evaluation_count(result),
            "seconds": seconds,
            "seed_transfer_gap": float(diagnostics.seed_transfer_gap),
            "seed_right_fixed_point_minimum_eigenvalue": float(
                diagnostics.seed_right_fixed_point_minimum_eigenvalue
            ),
            "selected": False,
        }
        rows.append(row)
        print(
            f"embedding-screen seed={candidate_seed} cost={result.cost:.2e} "
            f"status={result.status} evals={optimizer_evaluation_count(result)}",
            flush=True,
        )
        if best is None or result.cost < best[0]:
            best = (float(result.cost), seed_tensor, diagnostics)

    if best is None:
        raise ValueError("--embedding-candidate-seeds did not contain any seeds")
    selected_seed = int(best[2].seed)
    for row in rows:
        row["selected"] = row["seed"] == selected_seed
    print(f"embedding-screen selected_seed={selected_seed}", flush=True)
    return best[1], best[2], rows


def select_external_initial_seed(
    seed_source: np.ndarray,
    trajectory_bond_dimension: int,
    *,
    protocol: LocalEvolutionProtocol,
    lift_noise_amplitude: float,
    embedding_seed: int,
    canonical_tolerance: float,
) -> tuple[np.ndarray, InitialLiftDiagnostics]:
    """Use an explicit optimizer seed, lifting it to the trajectory D if needed."""

    return lift_left_canonical_seed(
        seed_source,
        trajectory_bond_dimension,
        noise_amplitude=lift_noise_amplitude,
        seed=embedding_seed,
        diagnostic_block_length=protocol.block_length,
        canonical_tolerance=canonical_tolerance,
    )


def fit_fixed_target(
    seed: np.ndarray,
    target: np.ndarray,
    primary: OptimizerOptions,
    strict: OptimizerOptions,
    seed_fixed_point: np.ndarray | None = None,
    *,
    use_strict_retry: bool = True,
) -> tuple[np.ndarray, object, bool, float]:
    """Fit one seed to an unchanged target, retrying strictly if needed."""

    started = time.perf_counter()
    block_length = infer_block_length(seed, target)
    best_A, best = run_fixed_target_optimizer(
        seed,
        target,
        block_length,
        primary,
        initial_fixed_point=seed_fixed_point,
    )
    used_strict = False
    if use_strict_retry and best.cost > primary.cost_tolerance:
        # Continue from the best primary iterate.  Restarting the strict solve
        # from the original seed discards precisely the progress that the
        # retry is meant to refine, and is especially damaging when a solve
        # has stopped only marginally above the acceptance threshold.
        best_r, _ = optimizer_right_fixed_point(
            best_A,
            strict.fixed_point_solver,
        )
        strict_A, strict_result = run_fixed_target_optimizer(
            best_A,
            target,
            block_length,
            strict,
            initial_fixed_point=best_r,
        )
        if strict_result.cost < best.cost:
            best_A, best = strict_A, strict_result
            used_strict = True
    return best_A, best, used_strict, time.perf_counter() - started


def fit_first_step_with_cg(
    seed: np.ndarray,
    target: np.ndarray,
    cg_options: CGOptions,
) -> tuple[np.ndarray, object, float]:
    """Fit the first lifted target with analytic CG before LM polishing."""

    started = time.perf_counter()
    block_length = infer_block_length(seed, target)
    A, result = optimize_fixed_target_cg(seed, target, block_length, cg_options)
    return A, result, time.perf_counter() - started


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
    if args.stop_after_step < 0 or args.stop_after_step > args.steps:
        raise ValueError("--stop-after-step must be zero or lie within --steps")
    if args.maximum_step_seconds < 0:
        raise ValueError("--maximum-step-seconds must be non-negative")
    if args.runtime_promotion_factor < 0:
        raise ValueError("--runtime-promotion-factor must be non-negative")
    if args.runtime_promotion_window < 1:
        raise ValueError("--runtime-promotion-window must be positive")
    if args.runtime_promotion_minimum_seconds < 0:
        raise ValueError(
            "--runtime-promotion-minimum-seconds must be non-negative"
        )
    if args.accept_cost <= 0:
        raise ValueError("--accept-cost must be positive")
    first_step_accept_cost = effective_first_step_accept_cost(
        args.first_step_accept_cost,
        args.accept_cost,
    )
    if first_step_accept_cost <= 0:
        raise ValueError("--first-step-accept-cost must be positive")
    if args.first_step_cg_max_iterations < 0 or args.first_step_cg_restart < 1:
        raise ValueError("invalid first-step CG iteration controls")
    if args.optimizer_iteration_cap < 0 or args.first_step_lm_iteration_cap < 0:
        raise ValueError("LM iteration caps must be non-negative")
    if args.lm_jacobian_workers < 1:
        raise ValueError("--lm-jacobian-workers must be positive")
    if args.matrix_free_dense_rescue_iteration_cap < 0:
        raise ValueError("dense-rescue LM iteration cap must be non-negative")
    if args.matrix_free_dense_rescue and args.optimizer != "matrix-free-lm":
        raise ValueError("dense rescue requires --optimizer matrix-free-lm")
    if args.first_step_cg_seconds < 0:
        raise ValueError("--first-step-cg-seconds must be non-negative")
    if (
        args.optimizer_max_seconds <= 0
        or args.first_step_lm_seconds <= 0
        or args.matrix_free_dense_rescue_seconds <= 0
    ):
        raise ValueError("optimizer wall-time caps must be positive")
    if args.horizontal_cg_restart < 1:
        raise ValueError("--horizontal-cg-restart must be positive")
    if args.horizontal_cg_maximum_line_search_evaluations < 1:
        raise ValueError(
            "--horizontal-cg-maximum-line-search-evaluations must be positive"
        )
    if args.horizontal_cg_adjoint_rtol <= 0:
        raise ValueError("--horizontal-cg-adjoint-rtol must be positive")
    if not (
        1
        <= args.matrix_free_krylov_initial_iterations
        <= args.matrix_free_krylov_max_iterations
    ):
        raise ValueError(
            "matrix-free Krylov iteration limits must satisfy "
            "1 <= initial <= maximum"
        )
    if not (
        0 < args.matrix_free_krylov_minimum_relative_tolerance
        <= args.matrix_free_krylov_relative_tolerance
    ):
        raise ValueError(
            "matrix-free Krylov tolerances must satisfy "
            "0 < minimum <= initial"
        )
    if args.matrix_free_adjoint_rtol <= 0 or args.matrix_free_jvp_rtol <= 0:
        raise ValueError("matrix-free fixed-point tolerances must be positive")
    if args.matrix_free_lsmr_condition_limit <= 1:
        raise ValueError("--matrix-free-lsmr-condition-limit must exceed one")
    if args.matrix_free_fixed_point_response_max_mb < 0:
        raise ValueError(
            "--matrix-free-fixed-point-response-max-mb must be non-negative"
        )
    if (
        args.initial_seed_lift_noise_amplitude is not None
        and args.initial_seed_lift_noise_amplitude <= 0
    ):
        raise ValueError("--initial-seed-lift-noise-amplitude must be positive")
    protocol = protocol_from_args(args)
    amplitudes = tuple(float(value) for value in args.perturb_amplitudes.split(","))
    if not amplitudes or args.perturbations_per_amplitude < 0 or args.random_restarts < 0:
        raise ValueError("invalid restart counts or amplitudes")
    initial_source, initial_source_input = load_tensor_file(
        args.initial_A,
        key=args.initial_key,
        layout=args.initial_layout,
    )
    if args.initial_seed_A is None:
        initial_seed_source = None
        initial_seed_source_input = None
    else:
        initial_seed_source, initial_seed_source_input = load_tensor_file(
            args.initial_seed_A,
            key=args.initial_seed_key,
            layout=args.initial_seed_layout,
        )
    trajectory_bond_dimension = (
        (
            initial_seed_source.shape[1]
            if initial_seed_source is not None
            else initial_source.shape[1]
        )
        if args.bond_dimension <= 0
        else args.bond_dimension
    )
    initial_seed_lift_noise_amplitude = effective_initial_seed_lift_noise(
        requested=args.initial_seed_lift_noise_amplitude,
        trajectory_bond_dimension=trajectory_bond_dimension,
        embedding_noise_amplitude=args.embedding_noise_amplitude,
    )
    if args.optimizer == "matrix-free-lm":
        optimizer_option_arguments = {
            "rank_tolerance": args.rank_tolerance,
            "fixed_point_solver": args.fixed_point_solver,
            "krylov_initial_iterations": (
                args.matrix_free_krylov_initial_iterations
            ),
            "krylov_max_iterations": args.matrix_free_krylov_max_iterations,
            "krylov_relative_tolerance": (
                args.matrix_free_krylov_relative_tolerance
            ),
            "krylov_minimum_relative_tolerance": (
                args.matrix_free_krylov_minimum_relative_tolerance
            ),
            "adaptive_krylov_tolerance": (
                args.matrix_free_adaptive_krylov_tolerance
            ),
            "adjoint_rtol": args.matrix_free_adjoint_rtol,
            "jvp_fixed_point_rtol": args.matrix_free_jvp_rtol,
            "verbose": args.matrix_free_verbose,
            "krylov_solver": args.matrix_free_krylov_solver,
            "krylov_preconditioner": (
                args.matrix_free_krylov_preconditioner.replace("-", "_")
            ),
            "recycle_krylov_solution": (
                args.matrix_free_recycle_krylov_solution
            ),
            "lsmr_condition_limit": (
                args.matrix_free_lsmr_condition_limit
            ),
            "fixed_point_response_solver": (
                args.matrix_free_fixed_point_response_solver.replace("-", "_")
            ),
            "fixed_point_response_maximum_bytes": int(
                args.matrix_free_fixed_point_response_max_mb * 1024**2
            ),
        }
        primary, strict = matrix_free_options(
            args.accept_cost,
            maximum_seconds=args.optimizer_max_seconds,
            iteration_cap=args.optimizer_iteration_cap,
            **optimizer_option_arguments,
        )
        first_primary, first_strict = matrix_free_options(
            first_step_accept_cost,
            maximum_seconds=args.first_step_lm_seconds,
            iteration_cap=args.first_step_lm_iteration_cap,
            **optimizer_option_arguments,
        )
    elif args.optimizer == "horizontal-cg":
        horizontal_option_arguments = {
            "fixed_point_solver": args.fixed_point_solver,
            "preconditioner": args.horizontal_cg_preconditioner.replace(
                "-", "_"
            ),
            "conjugacy_metric": args.horizontal_cg_conjugacy_metric.replace(
                "-", "_"
            ),
            "line_search_interpolation": (
                args.horizontal_cg_line_interpolation
            ),
            "restart": args.horizontal_cg_restart,
            "maximum_line_search_evaluations": (
                args.horizontal_cg_maximum_line_search_evaluations
            ),
            "adjoint_rtol": args.horizontal_cg_adjoint_rtol,
            "gauge_projector": args.horizontal_cg_gauge_projector,
            "verbose": args.horizontal_cg_verbose,
        }
        primary, strict = horizontal_cg_options(
            args.accept_cost,
            maximum_seconds=args.optimizer_max_seconds,
            iteration_cap=args.optimizer_iteration_cap,
            **horizontal_option_arguments,
        )
        first_primary, first_strict = horizontal_cg_options(
            first_step_accept_cost,
            maximum_seconds=args.first_step_lm_seconds,
            iteration_cap=args.first_step_lm_iteration_cap,
            **horizontal_option_arguments,
        )
    else:
        primary, strict = options(
            args.accept_cost,
            args.rank_tolerance,
            args.fixed_point_solver,
            args.lm_linear_solver,
            args.optimizer_max_seconds,
            args.optimizer_iteration_cap,
            args.lm_tangent_slice.replace("-", "_"),
            args.lm_initial_damping,
            args.lm_rdm_vectorization,
            args.lm_jacobian_response_solver.replace("-", "_"),
            args.lm_jacobian_workers,
        )
        first_primary, first_strict = options(
            first_step_accept_cost,
            args.rank_tolerance,
            args.fixed_point_solver,
            args.lm_linear_solver,
            args.first_step_lm_seconds,
            args.first_step_lm_iteration_cap,
            args.lm_tangent_slice.replace("-", "_"),
            args.lm_initial_damping,
            args.lm_rdm_vectorization,
            args.lm_jacobian_response_solver.replace("-", "_"),
            args.lm_jacobian_workers,
        )
    dense_rescue_primary = None
    dense_rescue_strict = None
    if args.optimizer == "matrix-free-lm" and args.matrix_free_dense_rescue:
        dense_rescue_primary, dense_rescue_strict = options(
            args.accept_cost,
            args.rank_tolerance,
            args.fixed_point_solver,
            args.lm_linear_solver,
            args.matrix_free_dense_rescue_seconds,
            args.matrix_free_dense_rescue_iteration_cap,
            args.lm_tangent_slice.replace("-", "_"),
            args.lm_initial_damping,
            args.lm_rdm_vectorization,
            args.lm_jacobian_response_solver.replace("-", "_"),
            args.lm_jacobian_workers,
        )
    lifted_dense_primary = None
    lifted_dense_strict = None
    if args.lifted_first_step_backend == "dense-lm":
        lifted_dense_primary, lifted_dense_strict = options(
            first_step_accept_cost,
            args.rank_tolerance,
            args.fixed_point_solver,
            args.lm_linear_solver,
            args.first_step_lm_seconds,
            args.first_step_lm_iteration_cap,
            args.lm_tangent_slice.replace("-", "_"),
            args.lm_initial_damping,
            args.lm_rdm_vectorization,
            args.lm_jacobian_response_solver.replace("-", "_"),
            args.lm_jacobian_workers,
        )
    first_step_cg = first_step_cg_options(
        args,
        accept_cost=first_step_accept_cost,
        rank_tolerance=args.rank_tolerance,
        default_fixed_point_solver=args.fixed_point_solver,
    )
    embedding_candidate_seeds = parse_seed_list(args.embedding_candidate_seeds)
    policy = {
        "protocol": asdict(protocol),
        "lightcone_sites": protocol.lightcone_sites,
        "target_margins": [list(margin) for margin in protocol.target_margins],
        "fixed_target": True,
        "trajectory_bond_dimension": trajectory_bond_dimension,
        "initial_seed_A": (
            None if args.initial_seed_A is None else str(args.initial_seed_A.resolve())
        ),
        "initial_seed_lift_noise_amplitude": initial_seed_lift_noise_amplitude,
        "fixed_point_solver": args.fixed_point_solver,
        "lm_linear_solver": args.lm_linear_solver,
        "time_predictor": args.time_predictor,
        "accept_cost": args.accept_cost,
        "first_step_accept_cost": first_step_accept_cost,
        "initial_seed_mode": args.initial_seed_mode,
        "embedding_noise_amplitude": args.embedding_noise_amplitude,
        "embedding_seed": args.embedding_seed,
        "circuit_lift_mixing_amplitude": args.circuit_lift_mixing_amplitude,
        "circuit_lift_seed": args.circuit_lift_seed,
        "embedding_candidate_seeds": list(embedding_candidate_seeds),
        "embedding_screen_max_iterations": args.embedding_screen_max_iterations,
        "embedding_screen_seconds": args.embedding_screen_seconds,
        "embedding_screen_fixed_point_solver": args.embedding_screen_fixed_point_solver,
        "first_step_optimizer": args.first_step_optimizer,
        "first_step_cg": asdict(first_step_cg),
        "initial_canonical_tolerance": args.initial_canonical_tolerance,
        "perturb_amplitudes": list(amplitudes),
        "perturbations_per_amplitude": args.perturbations_per_amplitude,
        "random_restarts": args.random_restarts,
        "random_seed": args.random_seed,
        "strict_retry": args.strict_retry,
        "optimizer_iteration_cap": args.optimizer_iteration_cap,
        "first_step_lm_iteration_cap": args.first_step_lm_iteration_cap,
        "primary_optimizer": asdict(primary),
        "strict_optimizer": asdict(strict),
    }
    preserve_initial_source_history = (
        initial_seed_source is not None
        and initial_source.shape[1] == trajectory_bond_dimension
        and initial_source.shape[2] == trajectory_bond_dimension
    )
    if preserve_initial_source_history:
        policy["same_dimension_external_seed_preserves_source_history"] = True
    # Preserve the exact legacy dense-LM policy dictionary so existing dense
    # trajectories remain resumable after this optional backend was added.
    if args.optimizer != "dense-lm":
        policy["optimizer_backend"] = args.optimizer
    if args.matrix_free_dense_rescue:
        policy["matrix_free_dense_rescue"] = {
            "maximum_seconds": args.matrix_free_dense_rescue_seconds,
            "iteration_cap": args.matrix_free_dense_rescue_iteration_cap,
            "optimizer": asdict(dense_rescue_primary),
            "strict_optimizer": asdict(dense_rescue_strict),
        }
    if args.lifted_first_step_backend != "same":
        policy["lifted_first_step_backend"] = args.lifted_first_step_backend
    if args.runtime_promotion_factor > 0:
        policy["runtime_promotion"] = {
            "factor": args.runtime_promotion_factor,
            "window": args.runtime_promotion_window,
            "minimum_seconds": args.runtime_promotion_minimum_seconds,
        }
    if args.dense_verify_acceptance:
        policy["dense_verify_acceptance"] = True

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "states.npy"
    right_fixed_points_path = output / "right_fixed_points.npy"
    times_path = output / "times.npy"
    initial_source_path = output / "initial_source.npy"
    initial_source_right_fixed_point_path = (
        output / "initial_source_right_fixed_point.npy"
    )
    initial_seed_path = output / "initial_seed.npy"
    initial_seed_source_path = output / "initial_seed_source.npy"
    steps_path = output / "steps.csv"
    restarts_path = output / "restart_trials.csv"
    matrix_free_history_path = output / "matrix_free_optimizer_iterations.csv"
    horizontal_cg_history_path = output / "horizontal_cg_optimizer_iterations.csv"
    backend_rescues_path = output / "backend_rescue_trials.csv"
    metadata_path = output / "metadata.json"
    pause_path = output / "PAUSE"

    if args.resume:
        metadata = json.loads(metadata_path.read_text())
        if metadata["policy"] != policy or metadata["steps"] != args.steps:
            raise ValueError("run policy or requested length differs from checkpoint")
        completed = int(metadata["completed_steps"])
        states = np.lib.format.open_memmap(states_path, mode="r+")
        right_fixed_points = np.lib.format.open_memmap(
            right_fixed_points_path,
            mode="r+",
        )
        initial_source = np.load(initial_source_path, allow_pickle=False)
        if initial_source_right_fixed_point_path.exists():
            initial_source_r = np.load(
                initial_source_right_fixed_point_path,
                allow_pickle=False,
            )
        else:
            initial_source_r, _ = optimizer_right_fixed_point(
                initial_source,
                protocol.target_source_fixed_point_solver,
            )
        if completed == 0 and preserve_initial_source_history:
            A = np.load(initial_seed_path, allow_pickle=False)
            initial_optimizer_seed_r, _ = optimizer_right_fixed_point(
                A,
                args.fixed_point_solver,
            )
        else:
            A = states[completed].copy()
            initial_optimizer_seed_r = None
        metadata["status"] = "running"
        metadata["resumed_at"] = utc_now()
        metadata.pop("failure", None)
    else:
        if states_path.exists() or metadata_path.exists():
            raise FileExistsError("output exists; use --resume or a new output directory")
        if initial_seed_source is None:
            initial_seed, initial_lift, lift_screen = select_initial_seed(
                initial_source,
                trajectory_bond_dimension,
                protocol=protocol,
                primary=primary,
                initial_seed_mode=args.initial_seed_mode,
                embedding_noise_amplitude=args.embedding_noise_amplitude,
                embedding_seed=args.embedding_seed,
                embedding_candidate_seeds=embedding_candidate_seeds,
                embedding_screen_max_iterations=args.embedding_screen_max_iterations,
                embedding_screen_seconds=args.embedding_screen_seconds,
                embedding_screen_fixed_point_solver=args.embedding_screen_fixed_point_solver,
                circuit_lift_mixing_amplitude=args.circuit_lift_mixing_amplitude,
                circuit_lift_seed=args.circuit_lift_seed,
                canonical_tolerance=args.initial_canonical_tolerance,
            )
        else:
            initial_seed, initial_lift = select_external_initial_seed(
                initial_seed_source,
                trajectory_bond_dimension,
                protocol=protocol,
                lift_noise_amplitude=initial_seed_lift_noise_amplitude,
                embedding_seed=args.embedding_seed,
                canonical_tolerance=args.initial_canonical_tolerance,
            )
            lift_screen = []
        states = np.lib.format.open_memmap(
            states_path,
            mode="w+",
            dtype=np.complex128,
            shape=(args.steps + 1, *initial_seed.shape),
        )
        right_fixed_points = np.lib.format.open_memmap(
            right_fixed_points_path,
            mode="w+",
            dtype=np.complex128,
            shape=(args.steps + 1, initial_seed.shape[1], initial_seed.shape[2]),
        )
        initial_source_r, initial_source_r_info = optimizer_right_fixed_point(
            initial_source,
            protocol.target_source_fixed_point_solver,
        )
        initial_seed_r, initial_seed_r_info = optimizer_right_fixed_point(
            initial_seed,
            protocol.target_source_fixed_point_solver,
        )
        if preserve_initial_source_history:
            states[0] = initial_source
            right_fixed_points[0] = initial_source_r
            if args.fixed_point_solver == protocol.target_source_fixed_point_solver:
                initial_optimizer_seed_r = initial_seed_r
            else:
                initial_optimizer_seed_r, _ = optimizer_right_fixed_point(
                    initial_seed,
                    args.fixed_point_solver,
                )
        else:
            states[0] = initial_seed
            right_fixed_points[0] = initial_seed_r
            initial_optimizer_seed_r = None
        states.flush()
        right_fixed_points.flush()
        atomic_npy(initial_source_path, initial_source)
        atomic_npy(initial_source_right_fixed_point_path, initial_source_r)
        atomic_npy(initial_seed_path, initial_seed)
        if initial_seed_source is not None:
            atomic_npy(initial_seed_source_path, initial_seed_source)
        atomic_npy(
            times_path,
            args.base_time + protocol.delta_t * np.arange(args.steps + 1),
        )
        completed = 0
        A = initial_seed.copy()
        metadata = {
            "status": "running",
            "created_at": utc_now(),
            "protocol_preset": args.protocol,
            "initial_A": str(args.initial_A.resolve()),
            "initial_A_input": initial_source_input.to_json(),
            "initial_seed_A": (
                None
                if args.initial_seed_A is None
                else str(args.initial_seed_A.resolve())
            ),
            "initial_seed_A_input": (
                None
                if initial_seed_source_input is None
                else initial_seed_source_input.to_json()
            ),
            "initial_source_store": initial_source_path.name,
            "initial_source_right_fixed_point_store": (
                initial_source_right_fixed_point_path.name
            ),
            "initial_seed_store": initial_seed_path.name,
            "right_fixed_points_store": right_fixed_points_path.name,
            "initial_seed_source_store": (
                None if initial_seed_source is None else initial_seed_source_path.name
            ),
            "initial_source_shape": list(initial_source.shape),
            "initial_source_right_fixed_point_shape": list(initial_source_r.shape),
            "initial_seed_source_shape": (
                None if initial_seed_source is None else list(initial_seed_source.shape)
            ),
            "initial_seed_shape": list(initial_seed.shape),
            "states_zero_kind": (
                "initial_source"
                if preserve_initial_source_history
                else "initial_seed"
            ),
            "right_fixed_points_shape": list(right_fixed_points.shape),
            "initial_source_right_fixed_point_info": fixed_point_info_json(
                initial_source_r_info
            ),
            "initial_seed_right_fixed_point_info": fixed_point_info_json(
                initial_seed_r_info
            ),
            "initial_lift": asdict(initial_lift),
            "initial_lift_screen": lift_screen,
            "steps": args.steps,
            "base_time": args.base_time,
            "delta_t": protocol.delta_t,
            "final_time": args.base_time + args.steps * protocol.delta_t,
            "completed_steps": 0,
            "policy": policy,
        }
        atomic_json(metadata_path, metadata)

    preceding_step_seconds: list[float] = []
    if completed > 0 and steps_path.exists():
        with steps_path.open(newline="") as handle:
            saved_rows = list(csv.DictReader(handle))[:completed]
        preceding_step_seconds = [
            float(row["step_seconds"]) for row in saved_rows
        ]

    invocation_started = time.perf_counter()
    cumulative_seconds = float(metadata.get("cumulative_seconds", 0.0))
    rescue_events = int(metadata.get("rescue_events", 0))
    status = "completed"
    try:
        for step in range(completed + 1, args.steps + 1):
            if pause_path.exists():
                status = "paused_by_PAUSE_file"
                break
            if step_limit_reached(completed, args.stop_after_step):
                status = "paused_by_step_limit"
                break
            if args.run_time_limit > 0 and time.perf_counter() - invocation_started >= args.run_time_limit:
                status = "paused_by_run_time_limit"
                break
            started = time.perf_counter()
            # This target is computed exactly once.  Every rescue trial below
            # minimizes against this unchanged matrix.
            target_source = initial_source if step == 1 else A
            target_source_r = (
                initial_source_r
                if step == 1
                else np.asarray(right_fixed_points[completed], dtype=np.complex128)
            )
            target_source_kind = "initial_source" if step == 1 else "trajectory_state"
            target = protocol.target_rdm(target_source, target_source_r)
            use_first_step_cg = (
                args.first_step_optimizer == "cg-lm"
                and step == 1
                and initial_source.shape[1] < trajectory_bond_dimension
            )
            use_first_step_threshold = (
                step == 1 and initial_source.shape[1] < trajectory_bond_dimension
            )
            active_accept_cost = (
                first_step_accept_cost if use_first_step_threshold else args.accept_cost
            )
            active_primary = first_primary if use_first_step_threshold else primary
            active_strict = first_strict if use_first_step_threshold else strict
            if (
                use_first_step_threshold
                and lifted_dense_primary is not None
                and lifted_dense_strict is not None
            ):
                active_primary = lifted_dense_primary
                active_strict = lifted_dense_strict
            first_step_cg_result = None
            first_step_cg_seconds = 0.0
            warm_seed_cost = float("nan")
            predictor_seed_cost = float("nan")
            optimizer_seed_kind = "warm_start"
            predictor_fallback_used = False
            if use_first_step_cg:
                print(
                    "first-step optimizer=cg-lm target_source=initial_source "
                    f"source_D={initial_source.shape[1]} trajectory_D={trajectory_bond_dimension}",
                    flush=True,
                )
                cg_A, first_step_cg_result, first_step_cg_seconds = fit_first_step_with_cg(
                    A, target, first_step_cg
                )
                next_A, result, used_strict, polish_seconds = fit_fixed_target(
                    cg_A,
                    target,
                    active_primary,
                    active_strict,
                    use_strict_retry=args.strict_retry,
                )
                append_optimizer_history(
                    matrix_free_history_path,
                    horizontal_cg_history_path,
                    step=step,
                    attempt=(
                        "first_step_polish_strict"
                        if used_strict
                        else "first_step_polish"
                    ),
                    result=result,
                )
                fit_seconds = first_step_cg_seconds + polish_seconds
                warm_cost = float(first_step_cg_result.cost)
                print(
                    f"first-step cg cost={first_step_cg_result.cost:.2e} "
                    f"status={first_step_cg_result.status}; "
                    f"lm-polish cost={result.cost:.2e} status={result.status}",
                    flush=True,
                )
            else:
                optimizer_seed = A
                optimizer_seed_r = (
                    np.asarray(initial_optimizer_seed_r, dtype=np.complex128)
                    if step == 1 and initial_optimizer_seed_r is not None
                    else np.asarray(
                        right_fixed_points[completed], dtype=np.complex128
                    )
                )
                warm_seed_cost = 0.5 * float(
                    np.linalg.norm(
                        block_rdm(A, protocol.block_length, optimizer_seed_r) - target
                    )
                    ** 2
                )
                if args.time_predictor == "secant" and completed >= 1:
                    predicted_seed = trajectory_secant_predictor(
                        np.asarray(states[completed - 1], dtype=np.complex128), A
                    )
                    predicted_seed_r, _ = optimizer_right_fixed_point(
                        predicted_seed, args.fixed_point_solver
                    )
                    predictor_seed_cost = 0.5 * float(
                        np.linalg.norm(
                            block_rdm(
                                predicted_seed,
                                protocol.block_length,
                                predicted_seed_r,
                            )
                            - target
                        )
                        ** 2
                    )
                    if (
                        np.isfinite(predictor_seed_cost)
                        and predictor_seed_cost < warm_seed_cost
                    ):
                        optimizer_seed = predicted_seed
                        optimizer_seed_r = predicted_seed_r
                        optimizer_seed_kind = "trajectory_secant"
                next_A, result, used_strict, fit_seconds = fit_fixed_target(
                    optimizer_seed,
                    target,
                    active_primary,
                    active_strict,
                    seed_fixed_point=optimizer_seed_r,
                    use_strict_retry=args.strict_retry,
                )
                append_optimizer_history(
                    matrix_free_history_path,
                    horizontal_cg_history_path,
                    step=step,
                    attempt=(
                        f"{optimizer_seed_kind}_strict"
                        if used_strict
                        else optimizer_seed_kind
                    ),
                    result=result,
                )
                warm_cost = float(result.cost)
                if (
                    optimizer_seed_kind == "trajectory_secant"
                    and result.cost > active_accept_cost
                ):
                    fallback_A, fallback_result, fallback_strict, fallback_seconds = (
                        fit_fixed_target(
                            A,
                            target,
                            active_primary,
                            active_strict,
                            seed_fixed_point=np.asarray(
                                right_fixed_points[completed], dtype=np.complex128
                            ),
                            use_strict_retry=args.strict_retry,
                        )
                    )
                    fit_seconds += fallback_seconds
                    predictor_fallback_used = True
                    append_optimizer_history(
                        matrix_free_history_path,
                        horizontal_cg_history_path,
                        step=step,
                        attempt=(
                            "warm_start_fallback_strict"
                            if fallback_strict
                            else "warm_start_fallback"
                        ),
                        result=fallback_result,
                    )
                    if fallback_result.cost < result.cost:
                        next_A = fallback_A
                        result = fallback_result
                        used_strict = fallback_strict
                        optimizer_seed_kind = "warm_start_fallback"
            restart_used = False
            accepted_kind = (
                "first_step_cg_lm" if use_first_step_cg else optimizer_seed_kind
            )
            accepted_trial = -1
            trials = 0

            if (
                result.cost > active_accept_cost
                and not use_first_step_threshold
                and dense_rescue_primary is not None
                and dense_rescue_strict is not None
            ):
                matrix_free_cost = float(result.cost)
                rescue_A, rescue_result, rescue_strict, rescue_seconds = (
                    fit_fixed_target(
                        next_A,
                        target,
                        dense_rescue_primary,
                        dense_rescue_strict,
                        use_strict_retry=args.strict_retry,
                    )
                )
                fit_seconds += rescue_seconds
                append_csv(
                    backend_rescues_path,
                    {
                        "step": step,
                        "time": args.base_time + step * protocol.delta_t,
                        "source_backend": "matrix-free-lm",
                        "rescue_backend": "dense-lm",
                        "source_cost": matrix_free_cost,
                        "rescue_cost": rescue_result.cost,
                        "accept_cost": active_accept_cost,
                        "rescue_status": rescue_result.status,
                        "rescue_evaluations": optimizer_evaluation_count(
                            rescue_result
                        ),
                        "used_strict_retry": rescue_strict,
                        "seconds": rescue_seconds,
                        "accepted": rescue_result.cost <= active_accept_cost,
                    },
                )
                print(
                    f"backend-rescue step={step} matrix-free-cost="
                    f"{matrix_free_cost:.2e} dense-cost={rescue_result.cost:.2e} "
                    f"status={rescue_result.status}",
                    flush=True,
                )
                if rescue_result.cost < result.cost:
                    next_A, result = rescue_A, rescue_result
                    used_strict = rescue_strict
                    accepted_kind = "matrix_free_dense_rescue"
                if rescue_result.cost <= active_accept_cost:
                    rescue_events += 1

            if result.cost > active_accept_cost:
                for kind, amplitude, trial, seed, seed_distance in distant_seeds(
                    A,
                    step=step,
                    amplitudes=amplitudes,
                    per_amplitude=args.perturbations_per_amplitude,
                    random_restarts=args.random_restarts,
                    random_seed=args.random_seed,
                ):
                    candidate_A, candidate_result, candidate_strict, seconds = fit_fixed_target(
                        seed,
                        target,
                        active_primary,
                        active_strict,
                        use_strict_retry=args.strict_retry,
                    )
                    append_optimizer_history(
                        matrix_free_history_path,
                        horizontal_cg_history_path,
                        step=step,
                        attempt=(
                            f"restart_{kind}_{trial}_strict"
                            if candidate_strict
                            else f"restart_{kind}_{trial}"
                        ),
                        result=candidate_result,
                    )
                    trials += 1
                    append_csv(
                        restarts_path,
                        {
                            "step": step,
                            "time": args.base_time + step * protocol.delta_t,
                            "target_frozen": True,
                            "target_source": target_source_kind,
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
                            "evaluations": optimizer_evaluation_count(
                                candidate_result
                            ),
                            "used_strict_retry": candidate_strict,
                            "seconds": seconds,
                            "accept_cost": active_accept_cost,
                            "accepted": candidate_result.cost <= active_accept_cost,
                        },
                    )
                    print(
                        f"restart step={step} trial={trial} kind={kind} "
                        f"amplitude={amplitude:g} distance={seed_distance:.3f} "
                        f"cost={candidate_result.cost:.2e}",
                        flush=True,
                    )
                    if candidate_result.cost <= active_accept_cost:
                        next_A, result = candidate_A, candidate_result
                        restart_used = True
                        accepted_kind = kind
                        accepted_trial = trial
                        rescue_events += 1
                        break
                    if candidate_result.cost < result.cost:
                        next_A, result = candidate_A, candidate_result

            optimizer_internal_cost = float(result.cost)
            acceptance_r = None
            acceptance_r_info = None
            if args.dense_verify_acceptance:
                acceptance_r, acceptance_r_info = optimizer_right_fixed_point(
                    next_A,
                    "dense",
                )
                acceptance_difference = (
                    block_rdm(next_A, protocol.block_length, acceptance_r) - target
                )
                verified_cost = 0.5 * float(
                    np.vdot(acceptance_difference, acceptance_difference).real
                )
                result = replace(
                    result,
                    cost=verified_cost,
                    residual_norm=float(np.sqrt(2.0 * verified_cost)),
                )

            if result.cost > active_accept_cost:
                status = "failed_fixed_target_multistart"
                failed_r, failed_r_info = optimizer_right_fixed_point(
                    next_A,
                    protocol.target_source_fixed_point_solver,
                )
                failed_r_info_json = fixed_point_info_json(failed_r_info)
                failed_rdm = block_rdm(
                    next_A,
                    protocol.block_length,
                    failed_r,
                )
                failed_cost_recomputed = 0.5 * float(
                    np.linalg.norm(failed_rdm - target) ** 2
                )
                failed_stem = f"failed_best_step_{step:06d}"
                failed_tensor_path = output / f"{failed_stem}.npy"
                failed_fixed_point_path = output / f"{failed_stem}_rfp.npy"
                failed_target_path = output / f"failed_target_step_{step:06d}.npy"
                atomic_npy(failed_tensor_path, next_A)
                atomic_npy(failed_fixed_point_path, failed_r)
                atomic_npy(failed_target_path, target)
                metadata["failure"] = {
                    "attempted_step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "accept_cost": active_accept_cost,
                    "warm_start_cost": warm_cost,
                    "optimizer_seed_kind": optimizer_seed_kind,
                    "warm_seed_cost": warm_seed_cost,
                    "predictor_seed_cost": predictor_seed_cost,
                    "best_cost": result.cost,
                    "best_optimizer_internal_cost": optimizer_internal_cost,
                    "best_target_residual": result.residual_norm,
                    "best_cost_recomputed": failed_cost_recomputed,
                    "best_tensor_store": failed_tensor_path.name,
                    "best_right_fixed_point_store": failed_fixed_point_path.name,
                    "target_store": failed_target_path.name,
                    "best_optimizer_status": result.status,
                    "best_optimizer_evaluations": optimizer_evaluation_count(result),
                    "best_right_fixed_point_info": failed_r_info_json,
                    "best_left_canonical_error": canonical_errors(next_A)[
                        "left_canonical_error"
                    ],
                    "restart_trials": trials,
                }
                break

            A = next_A
            if acceptance_r is not None and acceptance_r_info is not None:
                A_r, A_r_info = acceptance_r, acceptance_r_info
            else:
                A_r, A_r_info = optimizer_right_fixed_point(
                    A,
                    protocol.target_source_fixed_point_solver,
                )
            A_r_info_json = fixed_point_info_json(A_r_info)
            step_seconds = time.perf_counter() - started
            states[step] = A
            right_fixed_points[step] = A_r
            completed = step
            cumulative_seconds += step_seconds
            runtime_diagnostic = runtime_promotion_diagnostic(
                step_seconds,
                preceding_step_seconds,
                factor=args.runtime_promotion_factor,
                window=args.runtime_promotion_window,
                minimum_seconds=args.runtime_promotion_minimum_seconds,
            )
            step_record = {
                    "step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "target_source": target_source_kind,
                    "target_source_bond_dimension": target_source.shape[1],
                    "warm_start_cost": warm_cost,
                    "optimizer_seed_kind": optimizer_seed_kind,
                    "warm_seed_cost": warm_seed_cost,
                    "predictor_seed_cost": predictor_seed_cost,
                    "predictor_fallback_used": predictor_fallback_used,
                    "first_step_cg_used": use_first_step_cg,
                    "first_step_cg_cost": (
                        float("nan")
                        if first_step_cg_result is None
                        else first_step_cg_result.cost
                    ),
                    "first_step_cg_target_residual": (
                        float("nan")
                        if first_step_cg_result is None
                        else first_step_cg_result.residual_norm
                    ),
                    "first_step_cg_status": (
                        ""
                        if first_step_cg_result is None
                        else first_step_cg_result.status
                    ),
                    "first_step_cg_evaluations": (
                        0
                        if first_step_cg_result is None
                        else optimizer_evaluation_count(first_step_cg_result)
                    ),
                    "first_step_cg_seconds": first_step_cg_seconds,
                    "optimizer_cost": result.cost,
                    "optimizer_internal_cost": optimizer_internal_cost,
                    "dense_acceptance_verified": args.dense_verify_acceptance,
                    "accept_cost": active_accept_cost,
                    "target_residual": result.residual_norm,
                    "optimizer_status": result.status,
                    "optimizer_evaluations": optimizer_evaluation_count(result),
                    "right_fixed_point_residual": A_r_info_json["residual"],
                    "right_fixed_point_min_eigenvalue": A_r_info_json["min_eigenvalue"],
                    "warm_used_strict_retry": used_strict,
                    "restart_used": restart_used,
                    "accepted_seed_kind": accepted_kind,
                    "accepted_restart_trial": accepted_trial,
                    "restart_trials": trials,
                    "fit_seconds_before_restarts": fit_seconds,
                    "step_seconds": step_seconds,
                    "left_canonical_error": canonical_errors(A)["left_canonical_error"],
            }
            if runtime_diagnostic is not None:
                step_record.update(
                    {
                        "runtime_reference_median_seconds": (
                            runtime_diagnostic["reference_median_seconds"]
                        ),
                        "runtime_ratio_to_reference": runtime_diagnostic["ratio"],
                        "runtime_promotion_triggered": runtime_diagnostic[
                            "triggered"
                        ],
                    }
                )
            append_csv(steps_path, step_record)
            preceding_step_seconds.append(step_seconds)
            if step % args.checkpoint_every == 0 or restart_used or step == args.steps:
                states.flush()
                right_fixed_points.flush()
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
            if (
                runtime_diagnostic is not None
                and runtime_diagnostic["triggered"]
            ):
                status = "promote_requested_by_runtime"
                metadata["runtime_promotion"] = {
                    "step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "step_seconds": step_seconds,
                    **runtime_diagnostic,
                }
                states.flush()
                right_fixed_points.flush()
                metadata.update(
                    {
                        "status": status,
                        "updated_at": utc_now(),
                        "completed_steps": completed,
                        "rescue_events": rescue_events,
                        "cumulative_seconds": cumulative_seconds,
                    }
                )
                atomic_json(metadata_path, metadata)
                print(
                    f"runtime-promotion step={step} "
                    f"time={args.base_time + step * protocol.delta_t:.3f} "
                    f"wall={step_seconds:.2f}s "
                    f"reference={runtime_diagnostic['reference_median_seconds']:.2f}s "
                    f"ratio={runtime_diagnostic['ratio']:.2f}",
                    flush=True,
                )
                break
            if step_runtime_limit_exceeded(
                step_seconds,
                args.maximum_step_seconds,
            ):
                status = "paused_by_step_runtime_limit"
                metadata["step_runtime_limit"] = {
                    "step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "step_seconds": step_seconds,
                    "maximum_step_seconds": args.maximum_step_seconds,
                }
                states.flush()
                right_fixed_points.flush()
                metadata.update(
                    {
                        "status": status,
                        "updated_at": utc_now(),
                        "completed_steps": completed,
                        "rescue_events": rescue_events,
                        "cumulative_seconds": cumulative_seconds,
                    }
                )
                atomic_json(metadata_path, metadata)
                print(
                    f"step-runtime-limit step={step} "
                    f"time={args.base_time + step * protocol.delta_t:.3f} "
                    f"wall={step_seconds:.2f}s "
                    f"limit={args.maximum_step_seconds:.2f}s",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        status = "paused_by_keyboard_interrupt"
    finally:
        states.flush()
        right_fixed_points.flush()
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
