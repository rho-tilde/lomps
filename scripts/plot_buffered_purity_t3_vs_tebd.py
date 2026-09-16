#!/usr/bin/env python3
"""Compare the t=3 strict-fibre L=4 tensor with ordinary LOMPS and TEBD."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lomps.buffered_purity import density_matrix_purity
from lomps.optimizer import LMOptions, optimize_tensor, optimizer_right_fixed_point
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm


Array = np.ndarray
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-A",
        type=Path,
        default=(
            ROOT
            / "data/production/yplus_l4_d12_t3p000_20260914/"
            "previous_t2p999_D12.npy"
        ),
    )
    parser.add_argument(
        "--old-lomps-A",
        type=Path,
        default=(
            ROOT
            / "data/production/yplus_l4_d12_t3p000_20260914/"
            "anchor_t3p000_D12.npy"
        ),
    )
    parser.add_argument("--fibre-history", type=Path, required=True)
    parser.add_argument(
        "--tebd-rho6",
        type=Path,
        default=Path(
            "/private/tmp/lomps_tebd_yplus_11218/"
            "all_rdms_L_rdm-6_11218.npy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "benchmarks/"
            "buffered_purity_l4_t3_tebd_comparison_20260914"
        ),
    )
    parser.add_argument("--time", type=float, default=3.0)
    return parser.parse_args()


def normalize(rho: Array) -> Array:
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho)


def reduce_contiguous(
    rho: Array,
    *,
    total_sites: int,
    start: int,
    length: int,
) -> Array:
    left = 2**start
    kept = 2**length
    right = 2 ** (total_sites - start - length)
    tensor = rho.reshape(left, kept, right, left, kept, right)
    return np.einsum("airajr->ij", tensor, optimize=True)


def tebd_translation_average(rho6: Array, length: int) -> Array:
    first = reduce_contiguous(
        rho6, total_sites=6, start=0, length=length
    )
    if length == 6:
        return normalize(first)
    second = reduce_contiguous(
        rho6, total_sites=6, start=1, length=length
    )
    return normalize(0.5 * (first + second))


def rdms(A: Array, maximum_length: int) -> list[Array]:
    right, _ = optimizer_right_fixed_point(A, "dense")
    return [normalize(block_rdm(A, length, right)) for length in range(1, maximum_length + 1)]


def distances(first: Array, second: Array) -> tuple[float, float]:
    difference = normalize(first) - normalize(second)
    difference = 0.5 * (difference + difference.conj().T)
    trace = 0.5 * float(np.abs(np.linalg.eigvalsh(difference)).sum())
    hilbert_schmidt = float(np.linalg.norm(difference))
    return trace, hilbert_schmidt


def ordinary_fit(previous: Array, anchor: Array) -> tuple[Array, object, LMOptions]:
    protocol = replace(
        NONINTEGRABLE_ISING,
        block_length=4,
        target_source_fixed_point_solver="dense",
    )
    target = protocol.target_rdm(previous)
    options = LMOptions(
        max_iterations=80,
        gradient_tolerance=1e-11,
        cost_tolerance=1e-17,
        rank_tolerance=1e-10,
        tangent_slice="grassmann",
        initial_damping=1e-10,
        maximum_seconds=600.0,
        fixed_point_solver="dense",
        linear_solver="normal",
        rdm_vectorization="hermitian",
        jacobian_response_solver="dense_lu",
        jacobian_workers=1,
        verbose=False,
    )
    fitted, result = optimize_tensor(anchor, target, 4, options)
    return fitted, result, options


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    previous = np.asarray(np.load(args.previous_A), dtype=np.complex128)
    old_trajectory = np.asarray(np.load(args.old_lomps_A), dtype=np.complex128)
    ordinary, fit_result, fit_options = ordinary_fit(previous, old_trajectory)
    with np.load(args.fibre_history) as history:
        if "W" not in history.files:
            raise ValueError("fibre history does not contain the terminal tensor W")
        fibre_isometry = np.asarray(history["W"], dtype=np.complex128)
        if fibre_isometry.shape != (24, 12):
            raise ValueError(
                "expected terminal fibre isometry shape (24, 12), got "
                f"{fibre_isometry.shape}"
            )
        fibre = fibre_isometry.reshape(2, 12, 12)
        history_summary = {
            key: history[key].item()
            for key in (
                "status",
                "accepted_steps",
                "initial_primary_cost",
                "final_primary_cost",
                "initial_purity",
                "final_purity",
                "final_null_gradient_norm",
                "final_relative_purity_drop",
            )
        }

    tebd_all = np.load(args.tebd_rho6, mmap_mode="r")
    if tebd_all.shape != (7002, 64, 64):
        raise ValueError(f"unexpected TEBD rho6 shape {tebd_all.shape}")
    tebd_index = int(round(args.time / 0.001))
    if abs(tebd_index * 0.001 - args.time) > 1e-12:
        raise ValueError("--time must lie on the dt=0.001 grid")
    tebd_rho6 = normalize(np.asarray(tebd_all[tebd_index]))

    ordinary_rdms = rdms(ordinary, 8)
    fibre_rdms = rdms(fibre, 8)
    trajectory_rdms = rdms(old_trajectory, 8)
    tebd_rdms = [tebd_translation_average(tebd_rho6, length) for length in range(1, 7)]

    ordinary_purity = np.array([density_matrix_purity(rho) for rho in ordinary_rdms])
    fibre_purity = np.array([density_matrix_purity(rho) for rho in fibre_rdms])
    tebd_purity = np.array([density_matrix_purity(rho) for rho in tebd_rdms])
    trace = np.empty((2, 6))
    hs = np.empty((2, 6))
    rows: list[dict[str, float | int]] = []
    for index, length in enumerate(range(1, 7)):
        trace[0, index], hs[0, index] = distances(
            ordinary_rdms[index], tebd_rdms[index]
        )
        trace[1, index], hs[1, index] = distances(
            fibre_rdms[index], tebd_rdms[index]
        )
        rows.append(
            {
                "length": length,
                "tebd_purity": tebd_purity[index],
                "ordinary_lomps_purity": ordinary_purity[index],
                "fibre_lomps_purity": fibre_purity[index],
                "ordinary_trace_distance_to_tebd": trace[0, index],
                "fibre_trace_distance_to_tebd": trace[1, index],
                "ordinary_hs_distance_to_tebd": hs[0, index],
                "fibre_hs_distance_to_tebd": hs[1, index],
            }
        )

    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    initial_p8 = ordinary_purity[7]
    final_p8 = fibre_purity[7]
    if not np.isclose(initial_p8, history_summary["initial_purity"], rtol=2e-10, atol=1e-13):
        raise RuntimeError("reproduced ordinary-fit P8 does not match the fibre ledger")
    if not np.isclose(final_p8, history_summary["final_purity"], rtol=2e-12, atol=1e-14):
        raise RuntimeError("terminal tensor P8 does not match the fibre ledger")

    trajectory_to_ordinary_rho4 = distances(
        trajectory_rdms[3], ordinary_rdms[3]
    )
    ordinary_to_fibre_rho4 = distances(
        ordinary_rdms[3], fibre_rdms[3]
    )
    summary = {
        "time": args.time,
        "protocol": "nonintegrable Ising Y+ quench",
        "lomps": {"block_length": 4, "bond_dimension": 12},
        "tebd_reference_maximum_length": 6,
        "tebd_rho8_available": False,
        "ordinary_fit": {
            "status": fit_result.status,
            "evaluations": len(fit_result.history),
            "options": asdict(fit_options),
        },
        "fibre_history": history_summary,
        "p8": {
            "ordinary_lomps": initial_p8,
            "fibre_lomps": final_p8,
            "relative_reduction": (initial_p8 - final_p8) / initial_p8,
            "maximally_mixed_lower_bound": 1.0 / 256.0,
        },
        "rho4_consistency": {
            "trajectory_to_refitted_trace_distance": trajectory_to_ordinary_rho4[0],
            "trajectory_to_refitted_hs_distance": trajectory_to_ordinary_rho4[1],
            "ordinary_to_fibre_trace_distance": ordinary_to_fibre_rho4[0],
            "ordinary_to_fibre_hs_distance": ordinary_to_fibre_rho4[1],
        },
        "trace_distance_ratio_fibre_over_ordinary": (trace[1] / trace[0]).tolist(),
        "hs_distance_ratio_fibre_over_ordinary": (hs[1] / hs[0]).tolist(),
        "sources": {
            "previous_A": str(args.previous_A.resolve()),
            "old_lomps_A": str(args.old_lomps_A.resolve()),
            "fibre_history": str(args.fibre_history.resolve()),
            "tebd_rho6": str(args.tebd_rho6.resolve()),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lengths = np.arange(1, 7)
    colors = {"tebd": "#202020", "ordinary": "#0072B2", "fibre": "#CC3311"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.semilogy(lengths, tebd_purity, "o-", color=colors["tebd"], label="TEBD")
    ax.semilogy(lengths, ordinary_purity[:6], "s--", color=colors["ordinary"], label="ordinary LOMPS")
    ax.semilogy(lengths, fibre_purity[:6], "D-.", color=colors["fibre"], label="fibre-selected LOMPS")
    ax.set(xlabel="block length $\ell$", ylabel=r"purity $\mathrm{Tr}(\rho_\ell^2)$", title="(a) Local purities")
    ax.set_xticks(lengths)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.semilogy(lengths, trace[0], "s--", color=colors["ordinary"], label="ordinary LOMPS")
    ax.semilogy(lengths, trace[1], "D-.", color=colors["fibre"], label="fibre-selected LOMPS")
    ax.set(xlabel="block length $\ell$", ylabel="trace distance to TEBD", title="(b) RDM trace-distance error")
    ax.set_xticks(lengths)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.semilogy(lengths, hs[0], "s--", color=colors["ordinary"], label="ordinary LOMPS")
    ax.semilogy(lengths, hs[1], "D-.", color=colors["fibre"], label="fibre-selected LOMPS")
    ax.set(xlabel="block length $\ell$", ylabel="Hilbert--Schmidt distance to TEBD", title="(c) RDM Hilbert--Schmidt error")
    ax.set_xticks(lengths)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    bars = ax.bar(
        [0, 1],
        [initial_p8, final_p8],
        color=[colors["ordinary"], colors["fibre"]],
        width=0.62,
    )
    ax.axhline(1.0 / 256.0, color="#666666", linestyle=":", linewidth=1.2, label="maximally mixed $1/256$")
    ax.set_xticks([0, 1], ["ordinary\nLOMPS", "fibre-selected\nLOMPS"])
    ax.set_ylabel(r"buffer purity $\mathrm{Tr}(\rho_8^2)$")
    ax.set_title("(d) Eight-site buffer purity")
    ax.bar_label(bars, labels=[f"{initial_p8:.5f}", f"{final_p8:.5f}"], padding=3)
    ax.text(
        0.5,
        0.72,
        f"{100.0 * summary['p8']['relative_reduction']:.1f}% reduction\nTEBD $\\rho_8$ unavailable",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle(r"Y$+$ quench at $t=3.000$: strict fixed-$\rho_4$ fibre test ($L=4,D=12$)")
    png = output / "t3_fibre_vs_ordinary_lomps_vs_tebd.png"
    pdf = output / "t3_fibre_vs_ordinary_lomps_vs_tebd.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
