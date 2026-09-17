#!/usr/bin/env python3
"""Plot paired ordinary/fixed-rho4-purity L=4 trajectories against TEBD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from lomps.buffered_purity import density_matrix_purity
from lomps.optimizer import optimizer_right_fixed_point
from lomps.rdm import block_rdm


Array = np.ndarray
STATE_PATTERN = re.compile(r"step_(\d+)_t([0-9]+\.[0-9]+)\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-run",
        type=Path,
        action="append",
        required=True,
        help="Control segment; repeat the option to stitch consecutive runs.",
    )
    parser.add_argument(
        "--purity-run",
        type=Path,
        action="append",
        required=True,
        help="Purity segment; repeat the option to stitch consecutive runs.",
    )
    parser.add_argument("--tebd-rho6", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(rho: Array) -> Array:
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho)


def trace_distance(first: Array, second: Array) -> float:
    difference = normalize(first) - normalize(second)
    difference = 0.5 * (difference + difference.conj().T)
    return 0.5 * float(np.abs(np.linalg.eigvalsh(difference)).sum())


def reduce_contiguous(
    rho: Array, *, total_sites: int, start: int, length: int
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


def indexed_states(runs: list[Path]) -> dict[int, tuple[float, Path]]:
    result: dict[int, tuple[float, Path]] = {}
    for run in runs:
        for path in sorted((run / "states").glob("step_*_t*.npy")):
            match = STATE_PATTERN.match(path.name)
            if match is None:
                continue
            time_value = float(match.group(2))
            time_index = int(round(time_value / 0.001))
            if abs(time_index * 0.001 - time_value) > 1e-12:
                raise ValueError(f"state time {time_value} is off the dt grid")
            result[time_index] = (time_value, path)
    if not result:
        raise FileNotFoundError(f"no saved states below {runs}")
    return result


def rdm_set(path: Path) -> dict[int, Array]:
    A = np.asarray(np.load(path), dtype=np.complex128)
    right, _ = optimizer_right_fixed_point(A, "dense")
    return {
        length: normalize(block_rdm(A, length, right))
        for length in (4, 5, 6, 8)
    }


def purity_ledger(runs: list[Path]) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for run in runs:
        path = run / "steps.csv"
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                time_value = float(row["time"])
                time_index = int(round(time_value / 0.001))
                result[time_index] = (
                    float(row["purity_before"]),
                    float(row["purity_after"]),
                )
    return result


def main() -> None:
    args = parse_args()
    control = indexed_states(args.control_run)
    purity = indexed_states(args.purity_run)
    steps = sorted(set(control) & set(purity))
    if not steps:
        raise RuntimeError("paired trajectories share no saved times")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tebd = np.load(args.tebd_rho6, mmap_mode="r")
    if tebd.ndim != 3 or tebd.shape[1:] != (64, 64):
        raise ValueError(f"unexpected TEBD rho6 shape {tebd.shape}")

    times = np.asarray([control[step][0] for step in steps])
    purity_times = np.asarray([purity[step][0] for step in steps])
    if not np.allclose(times, purity_times, atol=1e-12, rtol=0.0):
        raise RuntimeError("paired state times do not agree")

    p8 = {"control": np.empty(len(steps)), "purity": np.empty(len(steps))}
    distances = {
        branch: {length: np.empty(len(steps)) for length in (4, 5, 6)}
        for branch in ("control", "purity")
    }
    branch_distances = {
        length: np.empty(len(steps)) for length in (4, 5, 6)
    }
    rows: list[dict[str, float | int]] = []
    for index, (step, time_value) in enumerate(zip(steps, times, strict=True)):
        tebd_index = int(round(time_value / 0.001))
        if abs(tebd_index * 0.001 - time_value) > 1e-12:
            raise ValueError(f"state time {time_value} is off the TEBD grid")
        tebd6 = normalize(np.asarray(tebd[tebd_index]))
        tebd_rdms = {
            length: tebd_translation_average(tebd6, length)
            for length in (4, 5, 6)
        }
        branch_rdms: dict[str, dict[int, Array]] = {}
        for branch, states in (("control", control), ("purity", purity)):
            branch_rdms[branch] = rdm_set(states[step][1])
            p8[branch][index] = density_matrix_purity(
                branch_rdms[branch][8]
            )
            for length in (4, 5, 6):
                distances[branch][length][index] = trace_distance(
                    branch_rdms[branch][length], tebd_rdms[length]
                )
        for length in (4, 5, 6):
            branch_distances[length][index] = trace_distance(
                branch_rdms["control"][length],
                branch_rdms["purity"][length],
            )
        rows.append(
            {
                "trajectory_index": index,
                "absolute_time_index": step,
                "time": time_value,
                "control_p8": p8["control"][index],
                "purity_branch_p8": p8["purity"][index],
                **{
                    f"{branch}_rho{length}_trace_distance_to_tebd": (
                        distances[branch][length][index]
                    )
                    for branch in ("control", "purity")
                    for length in (4, 5, 6)
                },
                **{
                    f"control_to_purity_rho{length}_trace_distance": (
                        branch_distances[length][index]
                    )
                    for length in (4, 5, 6)
                },
            }
        )

    with (output / "trajectory_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ledger = purity_ledger(args.purity_run)
    before_times = np.asarray(
        [purity[step][0] for step in steps if step in ledger]
    )
    before_p8 = np.asarray([ledger[step][0] for step in steps if step in ledger])
    summary = {
        "shared_steps": len(steps) - 1,
        "initial_time": float(times[0]),
        "final_time": float(times[-1]),
        "control_final_p8": float(p8["control"][-1]),
        "purity_branch_final_p8": float(p8["purity"][-1]),
        "final_p8_ratio_purity_over_control": float(
            p8["purity"][-1] / p8["control"][-1]
        ),
        "final_trace_distances_to_tebd": {
            branch: {
                f"rho{length}": float(distances[branch][length][-1])
                for length in (4, 5, 6)
            }
            for branch in ("control", "purity")
        },
        "final_control_to_purity_trace_distances": {
            f"rho{length}": float(branch_distances[length][-1])
            for length in (4, 5, 6)
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    colors = {"control": "#0072B2", "purity": "#CC3311"}
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.1), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(times, p8["control"], color=colors["control"], label="standard LOMPS")
    ax.plot(times, p8["purity"], color=colors["purity"], label="purity-selected LOMPS")
    if len(before_times):
        ax.plot(
            before_times,
            before_p8,
            color="#777777",
            linewidth=0.9,
            alpha=0.8,
            label="purity branch before each fibre search",
        )
    ax.axhline(1.0 / 256.0, color="#555555", linestyle=":", linewidth=1.0)
    ax.set(ylabel=r"$P_8=\mathrm{Tr}(\rho_8^2)$", title="(a) Buffer purity")
    ax.legend(frameon=False, fontsize=8)

    for ax, length, label in zip(
        axes.flat[1:], (4, 5, 6), ("(b)", "(c)", "(d)"), strict=True
    ):
        for branch, branch_label, style in (
            ("control", "standard LOMPS", "--"),
            ("purity", "purity-selected LOMPS", "-"),
        ):
            ax.semilogy(
                times,
                distances[branch][length],
                linestyle=style,
                color=colors[branch],
                label=branch_label,
            )
        ax.semilogy(
            times,
            branch_distances[length],
            linestyle=":",
            color="#666666",
            label="standard to purity-selected",
        )
        ax.set(
            ylabel=fr"$D_{{\rm tr}}(\rho_{length},\rho_{length}^{{\rm TEBD}})$",
            title=f"{label} {length}-site RDM error",
        )
        ax.legend(frameon=False, fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("physical time $t$")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle(
        r"Y$+$, $L=4,D=12$: standard versus fixed-$\rho_4$ low-$P_8$ trajectory"
    )
    fig.savefig(output / "paired_trajectory_vs_tebd.png", dpi=220)
    fig.savefig(output / "paired_trajectory_vs_tebd.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        times,
        distances["control"][4],
        "--",
        color=colors["control"],
        label=r"$D_{\rm tr}(\rho_4^{\rm standard},\rho_4^{\rm TEBD})$",
    )
    axes[0].plot(
        times,
        distances["purity"][4],
        "-",
        color=colors["purity"],
        label=r"$D_{\rm tr}(\rho_4^{\rm purity},\rho_4^{\rm TEBD})$",
    )
    axes[0].set_ylabel("trace distance to TEBD")
    axes[0].set_title(r"Four-site RDM accuracy")
    axes[0].legend(frameon=False)
    axes[1].plot(
        times,
        branch_distances[4],
        color="#666666",
        label=r"$D_{\rm tr}(\rho_4^{\rm standard},\rho_4^{\rm purity})$",
    )
    axes[1].set(
        xlabel="physical time $t$",
        ylabel="inter-branch trace distance",
        title=r"Separation of the two $\rho_4$ trajectories",
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output / "rho4_trace_distances.png", dpi=220)
    fig.savefig(output / "rho4_trace_distances.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
