"""Initial-state loading and deterministic bond-dimension lifting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canonical import canonical_errors, stack_tensor, unstack_tensor
from .rdm import block_rdm
from .transfer import injectivity_diagnostics, right_fixed_point


Array = np.ndarray


@dataclass(frozen=True)
class InitialLiftDiagnostics:
    """Diagnostics for using a low-D source with a higher-D optimizer seed."""

    source_bond_dimension: int
    target_bond_dimension: int
    physical_dimension: int
    noise_amplitude: float
    seed: int
    source_left_canonical_error: float
    seed_left_canonical_error: float
    seed_transfer_gap: float
    seed_right_fixed_point_minimum_eigenvalue: float
    seed_rdm_frobenius_error_by_length: dict[str, float]


def product_tensor(vector: Array) -> Array:
    """Return the normalized product-state tensor for one physical vector."""

    vector = np.asarray(vector, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("product vector must be one-dimensional")
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("product vector must be nonzero")
    return (vector / norm).reshape(vector.shape[0], 1, 1)


def coerce_initial_tensor(value: Array) -> Array:
    """Convert supported initial-state arrays to ``(d,D,D)`` tensor form.

    Supported inputs are a product vector ``(d,)``, a tensor ``(d,D,D)``, or a
    singleton-batched tensor ``(1,d,D,D)``.
    """

    value = np.asarray(value, dtype=np.complex128)
    if value.ndim == 1:
        return product_tensor(value)
    if value.ndim == 4 and len(value) == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError(
            "initial state must be a product vector (d), A with shape "
            "(d,D,D), or a singleton batch (1,d,D,D)"
        )
    if value.shape[0] < 2:
        raise ValueError("physical dimension must be at least 2")
    return np.asarray(value, dtype=np.complex128)


def _phase_fixed_qr_tensor(A: Array) -> Array:
    d, D, _ = A.shape
    Q, R = np.linalg.qr(stack_tensor(A), mode="reduced")
    diagonal = np.diag(R)
    phases = np.ones(D, dtype=np.complex128)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return unstack_tensor(Q * phases.conj()[None, :], d, D)


def lift_left_canonical_seed(
    source: Array,
    target_bond_dimension: int,
    *,
    noise_amplitude: float = 1e-4,
    seed: int = 104_729,
    diagnostic_block_length: int = 1,
    canonical_tolerance: float = 1e-10,
) -> tuple[Array, InitialLiftDiagnostics]:
    """Lift a low-D left-canonical source to a deterministic high-D seed.

    The returned seed is only an optimizer starting point. The physical first
    target should still be built from ``source`` when the source bond dimension
    is smaller than the trajectory bond dimension.
    """

    source = coerce_initial_tensor(source)
    d, source_D, _ = source.shape
    target_D = int(target_bond_dimension)
    if target_D < source_D:
        raise ValueError("target_bond_dimension cannot be smaller than source D")
    if target_D < 1:
        raise ValueError("target_bond_dimension must be positive")
    if diagnostic_block_length < 1:
        raise ValueError("diagnostic_block_length must be positive")

    source_errors = canonical_errors(source)
    if source_errors["left_canonical_error"] > canonical_tolerance:
        raise ValueError(
            "initial tensor is not left-canonical within tolerance; "
            "canonicalize it before passing it to LOMPS"
        )

    if target_D == source_D:
        seed_tensor = source.copy()
    else:
        if noise_amplitude <= 0.0:
            raise ValueError("positive noise_amplitude is required when lifting")
        rng = np.random.default_rng(seed)
        seed_tensor = np.zeros((d, target_D, target_D), dtype=np.complex128)
        seed_tensor[:, :source_D, :source_D] = source
        noise = rng.normal(size=seed_tensor.shape) + 1j * rng.normal(
            size=seed_tensor.shape
        )
        mask = np.ones((target_D, target_D), dtype=bool)
        mask[:source_D, :source_D] = False
        seed_tensor += float(noise_amplitude) * noise * mask[None, :, :]
        seed_tensor = _phase_fixed_qr_tensor(seed_tensor)

    seed_errors = canonical_errors(seed_tensor)
    seed_r, fixed = right_fixed_point(seed_tensor)
    source_r, _ = right_fixed_point(source)
    rdm_errors = {}
    for length in range(1, diagnostic_block_length + 1):
        rdm_errors[str(length)] = float(
            np.linalg.norm(
                block_rdm(seed_tensor, length, seed_r)
                - block_rdm(source, length, source_r)
            )
        )
    diagnostics = InitialLiftDiagnostics(
        source_bond_dimension=source_D,
        target_bond_dimension=target_D,
        physical_dimension=d,
        noise_amplitude=0.0 if target_D == source_D else float(noise_amplitude),
        seed=int(seed),
        source_left_canonical_error=source_errors["left_canonical_error"],
        seed_left_canonical_error=seed_errors["left_canonical_error"],
        seed_transfer_gap=float(injectivity_diagnostics(seed_tensor)["gap"]),
        seed_right_fixed_point_minimum_eigenvalue=float(fixed["min_eigenvalue"]),
        seed_rdm_frobenius_error_by_length=rdm_errors,
    )
    return seed_tensor, diagnostics
