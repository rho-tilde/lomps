#!/usr/bin/env python3
"""Run capped two-site iTEBD for the non-integrable Ising Y+ quench.

The implementation uses the Vidal Gamma--Lambda form with a two-site unit
cell.  A second-order (even half, odd full, even half) product formula is
applied to

    h = - Z Z + 1.05 X I - 0.5 Z I,

matching the convention used by the LOMPS Y+ trajectories.  ``max_chi`` is
the Schmidt-rank cap on *each bond*; it is not multiplied by the number of
sites in the unit cell.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


Array = np.ndarray

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


@dataclass
class VidalState:
    """Two-site periodic infinite MPS in Gamma--Lambda form.

    ``lambda_a`` lies between A and B, while ``lambda_b`` lies between B and
    the A tensor of the next unit cell.
    """

    gamma_a: Array
    lambda_a: Array
    gamma_b: Array
    lambda_b: Array


def yplus_state() -> VidalState:
    vector = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    gamma = vector.reshape(1, 2, 1)
    lam = np.ones(1, dtype=np.float64)
    return VidalState(gamma.copy(), lam.copy(), gamma.copy(), lam.copy())


def ising_bond_hamiltonian() -> Array:
    return -np.kron(Z, Z) + 1.05 * np.kron(X, I2) - 0.5 * np.kron(Z, I2)


def unitary_from_hermitian(hamiltonian: Array, duration: float) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1.0j * float(duration) * eigenvalues)
    return (eigenvectors * phases[None, :]) @ eigenvectors.conj().T


def _safe_inverse(values: Array, relative_tolerance: float = 1.0e-14) -> Array:
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    threshold = relative_tolerance * max(scale, 1.0)
    inverse = np.zeros_like(values, dtype=np.float64)
    mask = np.abs(values) > threshold
    inverse[mask] = 1.0 / values[mask]
    return inverse


def update_bond(
    left: Array,
    centre: Array,
    right: Array,
    outer: Array,
    gate: Array,
    *,
    max_chi: int,
    svd_cutoff: float,
) -> tuple[Array, Array, Array, float]:
    """Apply a two-site gate and restore the Vidal decomposition."""

    theta = np.einsum(
        "a,aib,b,bjc,c->aijc",
        outer,
        left,
        centre,
        right,
        outer,
        optimize=True,
    )
    evolved = np.einsum(
        "pqij,aijc->apqc", gate.reshape(2, 2, 2, 2), theta, optimize=True
    )
    rows = len(outer) * 2
    matrix = evolved.reshape(rows, rows)
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)

    available = len(singular_values)
    keep = min(int(max_chi), available)
    if svd_cutoff > 0.0 and singular_values.size:
        # Absolute Schmidt-value threshold.  Schmidt values are normalized,
        # so this is also a stable convention across time steps.
        above = int(np.count_nonzero(singular_values > svd_cutoff))
        keep = max(1, min(keep, above))

    total_weight = float(np.sum(singular_values**2))
    discarded_weight = float(
        np.sum(singular_values[keep:] ** 2) / max(total_weight, 1.0e-300)
    )
    kept = np.asarray(singular_values[:keep], dtype=np.float64)
    kept /= np.linalg.norm(kept)

    outer_inverse = _safe_inverse(outer)
    new_left = u[:, :keep].reshape(len(outer), 2, keep)
    new_left = outer_inverse[:, None, None] * new_left
    new_right = vh[:keep].reshape(keep, 2, len(outer))
    new_right = new_right * outer_inverse[None, None, :]
    return new_left, kept, new_right, discarded_weight


def strang_step(
    state: VidalState,
    half_gate: Array,
    full_gate: Array,
    *,
    max_chi: int,
    svd_cutoff: float,
) -> tuple[float, float]:
    """Perform one even-half, odd-full, even-half iTEBD step."""

    discarded: list[float] = []

    state.gamma_a, state.lambda_a, state.gamma_b, weight = update_bond(
        state.gamma_a,
        state.lambda_a,
        state.gamma_b,
        state.lambda_b,
        half_gate,
        max_chi=max_chi,
        svd_cutoff=svd_cutoff,
    )
    discarded.append(weight)

    state.gamma_b, state.lambda_b, state.gamma_a, weight = update_bond(
        state.gamma_b,
        state.lambda_b,
        state.gamma_a,
        state.lambda_a,
        full_gate,
        max_chi=max_chi,
        svd_cutoff=svd_cutoff,
    )
    discarded.append(weight)

    state.gamma_a, state.lambda_a, state.gamma_b, weight = update_bond(
        state.gamma_a,
        state.lambda_a,
        state.gamma_b,
        state.lambda_b,
        half_gate,
        max_chi=max_chi,
        svd_cutoff=svd_cutoff,
    )
    discarded.append(weight)
    return float(sum(discarded)), float(max(discarded))


def origin_rdm(state: VidalState, length: int, origin: int) -> Array:
    """Return an exact finite-window RDM for origin A (0) or B (1)."""

    if length < 1:
        raise ValueError("length must be positive")
    if origin not in (0, 1):
        raise ValueError("origin must be 0 or 1")

    if origin == 0:
        tensors = (state.gamma_a, state.gamma_b)
        after = (state.lambda_a, state.lambda_b)
        left_lambda = state.lambda_b
    else:
        tensors = (state.gamma_b, state.gamma_a)
        after = (state.lambda_b, state.lambda_a)
        left_lambda = state.lambda_a

    amplitude = np.diag(left_lambda)
    for site in range(length):
        tensor = tensors[site % 2]
        right_lambda = after[site % 2]
        amplitude = np.einsum("...a,aib->...ib", amplitude, tensor, optimize=True)
        amplitude = amplitude * right_lambda

    # Axes are (left environment, physical sites..., right environment).
    physical_axes = tuple(range(1, length + 1))
    matrix = np.transpose(amplitude, physical_axes + (0, length + 1)).reshape(
        2**length, -1
    )
    rho = matrix @ matrix.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho)


def entropy(rho: Array) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    positive = eigenvalues[eigenvalues > 1.0e-15]
    return float(-np.sum(positive * np.log(positive)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chi", type=int, required=True)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--t-final", type=float, default=7.0)
    parser.add_argument("--sample-dt", type=float, default=0.01)
    parser.add_argument("--max-rdm-length", type=int, default=4)
    parser.add_argument("--svd-cutoff", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps_float = args.t_final / args.dt
    steps = int(round(steps_float))
    if not np.isclose(steps_float, steps, atol=1.0e-10, rtol=0.0):
        raise ValueError("t-final must be an integer multiple of dt")
    sample_stride_float = args.sample_dt / args.dt
    sample_stride = int(round(sample_stride_float))
    if not np.isclose(sample_stride_float, sample_stride, atol=1.0e-10, rtol=0.0):
        raise ValueError("sample-dt must be an integer multiple of dt")
    if args.max_chi < 1 or args.max_rdm_length < 1:
        raise ValueError("max-chi and max-rdm-length must be positive")

    state = yplus_state()
    bond_hamiltonian = ising_bond_hamiltonian()
    half_gate = unitary_from_hermitian(bond_hamiltonian, 0.5 * args.dt)
    full_gate = unitary_from_hermitian(bond_hamiltonian, args.dt)

    sample_steps = np.arange(0, steps + 1, sample_stride, dtype=np.int64)
    if sample_steps[-1] != steps:
        sample_steps = np.append(sample_steps, steps)
    sample_lookup = {int(step): index for index, step in enumerate(sample_steps)}
    samples: dict[str, Array] = {
        "times": sample_steps.astype(float) * args.dt,
        "chi_a": np.empty(len(sample_steps), dtype=np.int64),
        "chi_b": np.empty(len(sample_steps), dtype=np.int64),
        "x": np.empty(len(sample_steps), dtype=float),
        "y": np.empty(len(sample_steps), dtype=float),
        "z": np.empty(len(sample_steps), dtype=float),
        "energy": np.empty(len(sample_steps), dtype=float),
        "entropy_l1": np.empty(len(sample_steps), dtype=float),
    }
    for length in range(1, args.max_rdm_length + 1):
        dimension = 2**length
        samples[f"rho_l{length}"] = np.empty(
            (len(sample_steps), dimension, dimension), dtype=np.complex128
        )
        samples[f"rho_origin_a_l{length}"] = np.empty(
            (len(sample_steps), dimension, dimension), dtype=np.complex128
        )
        samples[f"rho_origin_b_l{length}"] = np.empty(
            (len(sample_steps), dimension, dimension), dtype=np.complex128
        )
        samples[f"origin_hs_l{length}"] = np.empty(len(sample_steps), dtype=float)

    discarded_sum = np.empty(steps, dtype=float)
    discarded_max = np.empty(steps, dtype=float)
    chi_a_history = np.empty(steps, dtype=np.int64)
    chi_b_history = np.empty(steps, dtype=np.int64)

    def record(step: int) -> None:
        index = sample_lookup[step]
        rho_by_length: dict[int, Array] = {}
        for length in range(1, args.max_rdm_length + 1):
            rho_a = origin_rdm(state, length, 0)
            rho_b = origin_rdm(state, length, 1)
            rho = 0.5 * (rho_a + rho_b)
            rho_by_length[length] = rho
            samples[f"rho_l{length}"][index] = rho
            samples[f"rho_origin_a_l{length}"][index] = rho_a
            samples[f"rho_origin_b_l{length}"][index] = rho_b
            samples[f"origin_hs_l{length}"][index] = np.linalg.norm(rho_a - rho_b)
        rho_one = rho_by_length[1]
        samples["chi_a"][index] = len(state.lambda_a)
        samples["chi_b"][index] = len(state.lambda_b)
        samples["x"][index] = float(np.trace(rho_one @ X).real)
        samples["y"][index] = float(np.trace(rho_one @ Y).real)
        samples["z"][index] = float(np.trace(rho_one @ Z).real)
        samples["entropy_l1"][index] = entropy(rho_one)
        rho_two = rho_by_length.get(2)
        if rho_two is None:
            rho_two = 0.5 * (origin_rdm(state, 2, 0) + origin_rdm(state, 2, 1))
        samples["energy"][index] = float(np.trace(rho_two @ bond_hamiltonian).real)

    started = time.perf_counter()
    record(0)
    for step in range(1, steps + 1):
        discarded_sum[step - 1], discarded_max[step - 1] = strang_step(
            state,
            half_gate,
            full_gate,
            max_chi=args.max_chi,
            svd_cutoff=args.svd_cutoff,
        )
        chi_a_history[step - 1] = len(state.lambda_a)
        chi_b_history[step - 1] = len(state.lambda_b)
        if step in sample_lookup:
            record(step)
            print(
                f"chi_cap={args.max_chi} step={step}/{steps} "
                f"t={step * args.dt:.6g} "
                f"chi=({len(state.lambda_a)},{len(state.lambda_b)}) "
                f"discarded={discarded_sum[step - 1]:.3e}",
                flush=True,
            )

    wall_seconds = time.perf_counter() - started
    metadata = {
        "algorithm": "two-site iTEBD in Vidal Gamma-Lambda form",
        "trotter_order": 2,
        "layer_order": "A-B half, B-A full, A-B half",
        "hamiltonian_bond": "-ZZ + 1.05 XI - 0.5 ZI",
        "initial_state": "Y+ product state",
        "max_chi_per_bond": args.max_chi,
        "dt": args.dt,
        "t_final": args.t_final,
        "sample_dt": args.sample_dt,
        "svd_cutoff_absolute_schmidt_value": args.svd_cutoff,
        "max_rdm_length": args.max_rdm_length,
        "window_combination": "arithmetic mean of A-origin and B-origin RDMs",
        "wall_seconds": wall_seconds,
    }
    samples.update(
        {
            "discarded_weight_sum": discarded_sum,
            "discarded_weight_max": discarded_max,
            "chi_a_history": chi_a_history,
            "chi_b_history": chi_b_history,
            "final_gamma_a": state.gamma_a,
            "final_lambda_a": state.lambda_a,
            "final_gamma_b": state.gamma_b,
            "final_lambda_b": state.lambda_b,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **samples)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output.resolve()} in {wall_seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main()
