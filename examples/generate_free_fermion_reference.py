#!/usr/bin/env python3
"""Generate an independent free-fermion reference for the integrable quench.

The transverse-field Ising model

    H(g) = -sum_j Z_j Z_(j+1) - g sum_j X_j

maps, after a Jordan-Wigner transformation, to a quadratic Majorana
Hamiltonian H=(i/4) w.T A w. A Gaussian state is therefore specified by its
Majorana covariance matrix Gamma. The ground-state covariance is obtained by
diagonalizing iA, and the post-quench covariance evolves as

    Gamma(t) = exp(A_1 t) Gamma(0) exp(A_1 t).T.

For a block of L sites, restricting Gamma to its 2L Majoranas gives

    Tr[rho_L(t) rho_L(0)]
        = 2**(-L) sqrt(det(I - Gamma_L(t) Gamma_L(0))).

The implementation of these steps lives in ``lomps.integrable``. This script
turns it into a reproducible dataset and performs a larger-ring convergence
check. It does not read a LOMPS trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lomps.integrable import exact_local_hs_rate, exact_transverse_magnetization


DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "integrable_tfim_reference"
    / "free_fermion_g0_1p5_g1_0p2_t0_t20.npz"
)
DEFAULT_PLOT = (
    ROOT / "docs" / "figures" / "integrable_tfim_free_fermion_reference.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0", type=float, default=1.5)
    parser.add_argument("--g1", type=float, default=0.2)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--delta-t", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--ring-sites", type=int, default=128)
    parser.add_argument(
        "--convergence-ring-sites",
        type=int,
        default=192,
        help="Larger ring used at sampled times to bound finite-ring effects.",
    )
    parser.add_argument("--convergence-points", type=int, default=41)
    parser.add_argument("--quadrature-points", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Defaults to the output path with a .json suffix.",
    )
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Generate only the NPZ and JSON files; matplotlib is then unnecessary.",
    )
    return parser.parse_args()


def inclusive_time_grid(final_time: float, delta_t: float) -> np.ndarray:
    if final_time < 0.0 or delta_t <= 0.0:
        raise ValueError("final_time must be non-negative and delta_t positive")
    steps = int(round(final_time / delta_t))
    if not np.isclose(steps * delta_t, final_time, atol=1e-12, rtol=1e-12):
        raise ValueError("final_time must be an integer multiple of delta_t")
    return delta_t * np.arange(steps + 1, dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_plot(
    path: Path,
    times: np.ndarray,
    sigma_x: np.ndarray,
    local_hs_rate: np.ndarray,
    *,
    g0: float,
    g1: float,
    patch_size: int,
    ring_sites: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for the plot; use --no-plot to generate "
            "the numerical reference without it"
        ) from error

    figure, (magnetization_axis, loschmidt_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        layout="constrained",
    )
    magnetization_axis.plot(times, sigma_x, color="#0072B2", linewidth=1.5)
    magnetization_axis.set_ylabel(r"$\langle X\rangle$")
    magnetization_axis.grid(True, alpha=0.22, linewidth=0.7)

    loschmidt_axis.plot(times, local_hs_rate, color="#D55E00", linewidth=1.5)
    loschmidt_axis.set(
        xlabel=r"Time $t$",
        ylabel=(
            rf"$-\frac{{1}}{{{patch_size}}}\log\,"
            rf"\mathrm{{Tr}}[\rho_{{{patch_size}}}(t)"
            rf"\rho_{{{patch_size}}}(0)]$"
        ),
    )
    loschmidt_axis.grid(True, alpha=0.22, linewidth=0.7)
    figure.suptitle(
        rf"Exact free-fermion TFIM reference: $g_0={g0:g}\to g_1={g1:g}$"
        rf", ring $N={ring_sites}$"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    times = inclusive_time_grid(args.final_time, args.delta_t)

    sigma_x = np.asarray(
        exact_transverse_magnetization(
            times,
            g0=args.g0,
            g1=args.g1,
            quadrature_points=args.quadrature_points,
        )
    )
    local_hs_rate = np.asarray(
        exact_local_hs_rate(
            times,
            args.patch_size,
            g0=args.g0,
            g1=args.g1,
            ring_sites=args.ring_sites,
        )
    )
    local_hs_overlap = np.exp(-args.patch_size * local_hs_rate)

    convergence_times = np.linspace(
        0.0,
        args.final_time,
        args.convergence_points,
    )
    reference_at_convergence_times = np.asarray(
        exact_local_hs_rate(
            convergence_times,
            args.patch_size,
            g0=args.g0,
            g1=args.g1,
            ring_sites=args.ring_sites,
        )
    )
    larger_ring_values = np.asarray(
        exact_local_hs_rate(
            convergence_times,
            args.patch_size,
            g0=args.g0,
            g1=args.g1,
            ring_sites=args.convergence_ring_sites,
        )
    )
    ring_convergence_error = float(
        np.max(np.abs(reference_at_convergence_times - larger_ring_values))
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        times=times,
        sigma_x=sigma_x,
        sigma_y=np.zeros_like(times),
        sigma_z=np.zeros_like(times),
        local_hs_rate=local_hs_rate,
        local_hs_overlap=local_hs_overlap,
    )

    metadata_path = (
        args.metadata_output
        if args.metadata_output is not None
        else args.output.with_suffix(".json")
    )
    metadata = {
        "format_version": 1,
        "description": (
            "Independent free-fermion reference for the integrable TFIM quench."
        ),
        "initial_hamiltonian": (
            f"H0 = -sum_j Z_j Z_(j+1) - {args.g0:g} sum_j X_j"
        ),
        "evolution_hamiltonian": (
            f"H1 = -sum_j Z_j Z_(j+1) - {args.g1:g} sum_j X_j"
        ),
        "parameters": {
            "g0": args.g0,
            "g1": args.g1,
            "final_time": args.final_time,
            "delta_t": args.delta_t,
            "patch_size": args.patch_size,
            "ring_sites": args.ring_sites,
            "quadrature_points": args.quadrature_points,
        },
        "arrays": {
            "times": "Inclusive uniform time grid.",
            "sigma_x": "Exact thermodynamic-limit transverse magnetization.",
            "sigma_y": "Exactly zero by the Ising parity symmetry of this quench.",
            "sigma_z": "Exactly zero by the Ising parity symmetry of this quench.",
            "local_hs_rate": (
                "-log(Tr[rho_L(t) rho_L(0)]) / L for the configured patch."
            ),
            "local_hs_overlap": "Tr[rho_L(t) rho_L(0)].",
        },
        "finite_ring_check": {
            "comparison_ring_sites": args.convergence_ring_sites,
            "sampled_times": args.convergence_points,
            "maximum_absolute_local_hs_rate_difference": ring_convergence_error,
        },
        "implementation": {
            "generator_script": str(Path(__file__).relative_to(ROOT)),
            "free_fermion_module": "src/lomps/integrable.py",
            "generator_script_sha256": sha256(Path(__file__)),
            "free_fermion_module_sha256": sha256(
                ROOT / "src" / "lomps" / "integrable.py"
            ),
        },
        "npz_sha256": sha256(args.output),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if not args.no_plot:
        save_plot(
            args.plot,
            times,
            sigma_x,
            local_hs_rate,
            g0=args.g0,
            g1=args.g1,
            patch_size=args.patch_size,
            ring_sites=args.ring_sites,
        )

    print(f"Saved data: {args.output}")
    print(f"Saved metadata: {metadata_path}")
    if not args.no_plot:
        print(f"Saved plot: {args.plot}")
    print(
        "Finite-ring check: "
        f"N={args.ring_sites} versus N={args.convergence_ring_sites}, "
        f"max difference={ring_convergence_error:.3e}"
    )


if __name__ == "__main__":
    main()
