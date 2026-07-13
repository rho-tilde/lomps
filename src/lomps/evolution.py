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
    coerce_initial_tensor,
    lift_left_canonical_seed,
    product_circuit_left_canonical_seed,
)
from .fixed_target_cg import optimize_fixed_target_cg
from .optimizer import CGOptions, LMOptions, optimizer_right_fixed_point, optimize_tensor
from .protocol import LocalEvolutionProtocol, NONINTEGRABLE_ISING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-A", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--base-time", type=float, required=True)
    parser.add_argument("--block-length", type=int, default=NONINTEGRABLE_ISING.block_length)
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
    parser.add_argument("--accept-cost", type=float, default=3e-16)
    parser.add_argument(
        "--first-step-accept-cost",
        type=float,
        default=None,
        help=(
            "Optional acceptance threshold only for the first low-D source to "
            "trajectory-D fit. Later trajectory steps still use --accept-cost."
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
        "--lm-linear-solver",
        choices=("normal", "svd"),
        default="normal",
        help=(
            "Linear solver inside LM optimizer evaluations. 'normal' builds "
            "the dense Jacobian but solves damped normal equations by Cholesky; "
            "'svd' keeps the older SVD-LM step and rank diagnostics."
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
        default=1e-8,
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


def load_initial(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    return coerce_initial_tensor(value)


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
) -> tuple[LMOptions, LMOptions]:
    primary = LMOptions(
        max_iterations=40_000,
        gradient_tolerance=1e-11,
        cost_tolerance=accept_cost,
        rank_tolerance=rank_tolerance,
        fixed_point_solver=fixed_point_solver,
        linear_solver=linear_solver,
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
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    if args.odd_parity_warning_threshold < 0:
        raise ValueError("--odd-parity-warning-threshold must be non-negative")
    protocol = replace(
        NONINTEGRABLE_ISING,
        name=f"nonintegrable_ising_L{args.block_length}",
        block_length=args.block_length,
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
    primary: LMOptions,
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
        _, result = optimize_tensor(
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
            "evaluations": len(result.history),
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
            f"status={result.status} evals={len(result.history)}",
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
    primary: LMOptions,
    strict: LMOptions,
    seed_fixed_point: np.ndarray | None = None,
    *,
    use_strict_retry: bool = True,
) -> tuple[np.ndarray, object, bool, float]:
    """Fit one seed to an unchanged target, retrying strictly if needed."""

    started = time.perf_counter()
    block_length = infer_block_length(seed, target)
    best_A, best = optimize_tensor(
        seed,
        target,
        block_length,
        primary,
        initial_fixed_point=seed_fixed_point,
    )
    used_strict = False
    if use_strict_retry and best.cost > primary.cost_tolerance:
        strict_A, strict_result = optimize_tensor(
            seed,
            target,
            block_length,
            strict,
            initial_fixed_point=seed_fixed_point,
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
    if args.first_step_cg_seconds < 0:
        raise ValueError("--first-step-cg-seconds must be non-negative")
    if (
        args.initial_seed_lift_noise_amplitude is not None
        and args.initial_seed_lift_noise_amplitude <= 0
    ):
        raise ValueError("--initial-seed-lift-noise-amplitude must be positive")
    protocol = protocol_from_args(args)
    amplitudes = tuple(float(value) for value in args.perturb_amplitudes.split(","))
    if not amplitudes or args.perturbations_per_amplitude < 0 or args.random_restarts < 0:
        raise ValueError("invalid restart counts or amplitudes")
    initial_source = load_initial(args.initial_A)
    initial_seed_source = (
        None if args.initial_seed_A is None else load_initial(args.initial_seed_A)
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
    primary, strict = options(
        args.accept_cost,
        args.rank_tolerance,
        args.fixed_point_solver,
        args.lm_linear_solver,
    )
    first_primary, first_strict = options(
        first_step_accept_cost,
        args.rank_tolerance,
        args.fixed_point_solver,
        args.lm_linear_solver,
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
        "primary_optimizer": asdict(primary),
        "strict_optimizer": asdict(strict),
    }

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
        A = states[completed].copy()
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
        states[0] = initial_seed
        right_fixed_points[0] = initial_seed_r
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
            "initial_A": str(args.initial_A.resolve()),
            "initial_seed_A": (
                None
                if args.initial_seed_A is None
                else str(args.initial_seed_A.resolve())
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
            first_step_cg_result = None
            first_step_cg_seconds = 0.0
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
                fit_seconds = first_step_cg_seconds + polish_seconds
                warm_cost = float(first_step_cg_result.cost)
                print(
                    f"first-step cg cost={first_step_cg_result.cost:.2e} "
                    f"status={first_step_cg_result.status}; "
                    f"lm-polish cost={result.cost:.2e} status={result.status}",
                    flush=True,
                )
            else:
                next_A, result, used_strict, fit_seconds = fit_fixed_target(
                    A,
                    target,
                    active_primary,
                    active_strict,
                    seed_fixed_point=np.asarray(
                        right_fixed_points[completed],
                        dtype=np.complex128,
                    ),
                    use_strict_retry=args.strict_retry,
                )
                warm_cost = float(result.cost)
            restart_used = False
            accepted_kind = "first_step_cg_lm" if use_first_step_cg else "warm_start"
            accepted_trial = -1
            trials = 0

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
                            "evaluations": len(candidate_result.history),
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

            if result.cost > active_accept_cost:
                status = "failed_fixed_target_multistart"
                metadata["failure"] = {
                    "attempted_step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "accept_cost": active_accept_cost,
                    "warm_start_cost": warm_cost,
                    "best_cost": result.cost,
                    "best_target_residual": result.residual_norm,
                    "restart_trials": trials,
                }
                break

            A = next_A
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
            append_csv(
                steps_path,
                {
                    "step": step,
                    "time": args.base_time + step * protocol.delta_t,
                    "target_source": target_source_kind,
                    "target_source_bond_dimension": target_source.shape[1],
                    "warm_start_cost": warm_cost,
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
                        else len(first_step_cg_result.history)
                    ),
                    "first_step_cg_seconds": first_step_cg_seconds,
                    "optimizer_cost": result.cost,
                    "accept_cost": active_accept_cost,
                    "target_residual": result.residual_norm,
                    "optimizer_status": result.status,
                    "optimizer_evaluations": len(result.history),
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
                },
            )
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
