#!/usr/bin/env python
"""Audit an integrable LOMPS trajectory against exact and legacy references."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from lomps.canonical import canonical_errors
from lomps.gates import two_site_hamiltonian_tfim
from lomps.integrable import exact_local_hs_rate, exact_transverse_magnetization
from lomps.observables import PAULI_X, PAULI_Y, PAULI_Z, local_rdm, one_site_rdm
from lomps.optimizer import optimizer_right_fixed_point
from lomps.protocol import INTEGRABLE_TFIM, NONINTEGRABLE_ISING
from lomps.rdm import block_rdm
from lomps.tensor_io import load_tensor_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL = ROOT / "data" / "integrable_tfim_reference" / "tfim_g0_1p5_D12.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--legacy-trajectory", type=Path, default=None)
    parser.add_argument("--initial-A", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--g0", type=float, default=1.5)
    parser.add_argument("--quadrature-points", type=int, default=2048)
    parser.add_argument("--local-patch-size", type=int, default=4)
    parser.add_argument("--ring-sites", type=int, default=128)
    return parser.parse_args()


def reference_checks(initial_path: Path, legacy_path: Path | None) -> dict[str, object]:
    initial, input_info = load_tensor_file(initial_path)
    r, fp_info = optimizer_right_fixed_point(initial, "dense")
    tensor_target = INTEGRABLE_TFIM.target_rdm(initial, r)
    dense_target = replace(INTEGRABLE_TFIM, target_contraction="dense").target_rdm(
        initial, r
    )
    result: dict[str, object] = {
        "initial_input": input_info.to_json(),
        "initial_canonical": canonical_errors(initial),
        "initial_fixed_point": {
            key: float(np.real(value)) for key, value in fp_info.items()
        },
        "tensor_vs_dense_target_frobenius": float(
            np.linalg.norm(tensor_target - dense_target)
        ),
    }
    if legacy_path is None:
        return result

    with np.load(legacy_path, allow_pickle=False) as archive:
        legacy_state = np.asarray(archive["state"][0], dtype=np.complex128)
        legacy_time = float(archive["time"][0])
    legacy_A = np.transpose(legacy_state, (1, 0, 2))
    legacy_rho = block_rdm(legacy_A, INTEGRABLE_TFIM.block_length)
    candidates = {
        "integrable_asymmetric_g_minus_0p2": INTEGRABLE_TFIM,
        "integrable_symmetric_g_minus_0p2": replace(
            INTEGRABLE_TFIM, symmetric_transverse=True
        ),
        "integrable_asymmetric_wrong_sign_g_plus_0p2": replace(
            INTEGRABLE_TFIM, g=0.2
        ),
        "nonintegrable": NONINTEGRABLE_ISING,
    }
    result["legacy_first_state"] = {
        "path": str(legacy_path.resolve()),
        "time": legacy_time,
        "candidate_half_squared_frobenius_costs": {
            name: 0.5
            * float(np.linalg.norm(legacy_rho - protocol.target_rdm(initial, r)) ** 2)
            for name, protocol in candidates.items()
        },
    }
    return result


def trajectory_audit(
    run_dir: Path,
    *,
    g0: float,
    quadrature_points: int,
    local_patch_size: int,
    ring_sites: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    states = np.load(run_dir / "states.npy", mmap_mode="r")
    times = np.load(run_dir / "times.npy", mmap_mode="r")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("protocol_preset") != "integrable-tfim":
        raise ValueError("run metadata does not identify the integrable-tfim preset")
    run_protocol = metadata["policy"]["protocol"]
    if not np.isclose(run_protocol["J"], -1.0) or not np.isclose(
        run_protocol["h"], 0.0
    ):
        raise ValueError("exact TFIM audit requires J=-1 and h=0")
    completed = int(metadata["completed_steps"])
    states = states[: completed + 1]
    times = np.asarray(times[: completed + 1])
    fixed_points_path = run_dir / "right_fixed_points.npy"
    fixed_points = (
        np.load(fixed_points_path, mmap_mode="r")[: completed + 1]
        if fixed_points_path.exists()
        else None
    )

    h2 = two_site_hamiltonian_tfim(
        g=run_protocol["g"],
        h=run_protocol["h"],
        J=run_protocol["J"],
        symmetric_transverse=run_protocol["symmetric_transverse"],
    )
    sigma_x = np.empty(completed + 1)
    sigma_y = np.empty(completed + 1)
    sigma_z = np.empty(completed + 1)
    energy = np.empty(completed + 1)
    local_overlap = np.empty(completed + 1)
    initial_r = None if fixed_points is None else fixed_points[0]
    initial_rho = block_rdm(states[0], local_patch_size, initial_r)
    for index, A in enumerate(states):
        r = None if fixed_points is None else fixed_points[index]
        rho1 = one_site_rdm(A, r, solver="dense")
        rho2 = local_rdm(A, 2, r, solver="dense")
        sigma_x[index] = float(np.trace(rho1 @ PAULI_X).real)
        sigma_y[index] = float(np.trace(rho1 @ PAULI_Y).real)
        sigma_z[index] = float(np.trace(rho1 @ PAULI_Z).real)
        energy[index] = float(np.trace(rho2 @ h2).real)
        rho_local = block_rdm(A, local_patch_size, r)
        local_overlap[index] = float(np.trace(rho_local @ initial_rho).real)
    if np.any(local_overlap <= 0):
        raise FloatingPointError("local Hilbert--Schmidt overlap is non-positive")
    local_rate = -np.log(local_overlap) / local_patch_size
    exact_local_rate = np.asarray(
        exact_local_hs_rate(
            times,
            local_patch_size,
            g0=g0,
            g1=-float(run_protocol["g"]),
            ring_sites=ring_sites,
        )
    )
    local_rate_error = local_rate - exact_local_rate
    exact_x = np.asarray(
        exact_transverse_magnetization(
            times,
            g0=g0,
            g1=-float(run_protocol["g"]),
            quadrature_points=quadrature_points,
        )
    )
    x_error = sigma_x - exact_x
    arrays = {
        "times": times,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "energy_density": energy,
        "exact_sigma_x": exact_x,
        "sigma_x_error": x_error,
        "local_hs_overlap": local_overlap,
        "local_hs_rate": local_rate,
        "exact_local_hs_rate": exact_local_rate,
        "local_hs_rate_error": local_rate_error,
    }
    summary: dict[str, object] = {
        "run_dir": str(run_dir.resolve()),
        "protocol": run_protocol,
        "exact_quench_g0": float(g0),
        "completed_steps": completed,
        "first_time": float(times[0]),
        "last_time": float(times[-1]),
        "sigma_x_max_abs_error": float(np.max(np.abs(x_error))),
        "sigma_x_rms_error": float(np.sqrt(np.mean(x_error**2))),
        "local_patch_size": int(local_patch_size),
        "free_fermion_ring_sites": int(ring_sites),
        "local_hs_rate_max_abs_error": float(np.max(np.abs(local_rate_error))),
        "local_hs_rate_rms_error": float(np.sqrt(np.mean(local_rate_error**2))),
        "energy_initial": float(energy[0]),
        "energy_final": float(energy[-1]),
        "energy_max_drift": float(np.max(np.abs(energy - energy[0]))),
    }
    steps_path = run_dir / "steps.csv"
    if steps_path.exists():
        with steps_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        accepted_costs = [
            float(row.get("optimizer_cost", row.get("cost", "nan")))
            for row in rows[:completed]
        ]
        accepted_costs = [value for value in accepted_costs if np.isfinite(value)]
        summary["accepted_cost_max"] = (
            None if not accepted_costs else float(max(accepted_costs))
        )
    return summary, arrays


def main() -> None:
    args = parse_args()
    report = {
        "reference": reference_checks(args.initial_A, args.legacy_trajectory),
    }
    arrays: dict[str, np.ndarray] | None = None
    if args.run_dir is not None:
        report["trajectory"], arrays = trajectory_audit(
            args.run_dir,
            g0=args.g0,
            quadrature_points=args.quadrature_points,
            local_patch_size=args.local_patch_size,
            ring_sites=args.ring_sites,
        )
    print(json.dumps(report, indent=2))
    if args.output_prefix is not None:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.output_prefix.with_suffix(".json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if arrays is not None:
            np.savez_compressed(args.output_prefix.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
