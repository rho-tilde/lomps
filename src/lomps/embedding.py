"""Initial-state loading and deterministic bond-dimension lifting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from .canonical import canonical_errors, polar_retraction, stack_tensor, unstack_tensor
from .gates import partial_trace_sites
from .protocol import LocalEvolutionProtocol
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
    method: str = "embedding"
    base_bond_dimension: int | None = None
    seed_target_rdm_frobenius_error: float | None = None
    seed_target_trace_distance: float | None = None


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


def _open_product_mps(vector: Array, sites: int) -> list[Array]:
    vector = np.asarray(vector, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    return [vector.reshape(1, len(vector), 1).copy() for _ in range(sites)]


def _mps_norm_squared(tensors: list[Array]) -> float:
    environment = np.ones((1, 1), dtype=np.complex128)
    for tensor in tensors:
        environment = np.einsum(
            "ac,asb,csd->bd",
            environment,
            tensor,
            tensor.conj(),
            optimize=True,
        )
    return float(np.real_if_close(environment[0, 0]))


def _normalize_mps(tensors: list[Array]) -> None:
    norm = np.sqrt(max(_mps_norm_squared(tensors), 1e-300))
    tensors[0] = tensors[0] / norm


def _apply_two_site_gate(tensors: list[Array], bond: int, gate: Array) -> int:
    left = np.asarray(tensors[bond], dtype=np.complex128)
    right = np.asarray(tensors[bond + 1], dtype=np.complex128)
    Dl, d, middle = left.shape
    if right.shape[0] != middle or right.shape[1] != d:
        raise ValueError("adjacent MPS tensor dimensions do not match")
    Dr = right.shape[2]
    unitary = np.asarray(gate, dtype=np.complex128).reshape(d, d, d, d)
    theta = np.einsum("asm,mtb->astb", left, right, optimize=True)
    evolved = np.einsum("uvst,astb->auvb", unitary, theta, optimize=True)
    matrix = evolved.reshape(Dl * d, d * Dr)
    U, singular_values, Vh = la.svd(
        matrix,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    keep = len(singular_values)
    tensors[bond] = U[:, :keep].reshape(Dl, d, keep)
    tensors[bond + 1] = (
        singular_values[:keep, None] * Vh[:keep]
    ).reshape(keep, d, Dr)
    return keep


def _right_canonicalize_open_mps(tensors: list[Array]) -> None:
    for site in range(len(tensors) - 1, 0, -1):
        Dl, d, Dr = tensors[site].shape
        matrix = tensors[site].reshape(Dl, d * Dr)
        Q, R = np.linalg.qr(matrix.T, mode="reduced")
        new_left = Q.shape[1]
        tensors[site] = Q.T.reshape(new_left, d, Dr)
        tensors[site - 1] = np.einsum(
            "asb,bc->asc",
            tensors[site - 1],
            R.T,
            optimize=True,
        )


def _move_open_mps_centre_right(tensors: list[Array], site: int) -> None:
    Dl, d, Dr = tensors[site].shape
    Q, R = np.linalg.qr(tensors[site].reshape(Dl * d, Dr), mode="reduced")
    new_right = Q.shape[1]
    tensors[site] = Q.reshape(Dl, d, new_right)
    tensors[site + 1] = np.einsum(
        "ab,bsr->asr",
        R,
        tensors[site + 1],
        optimize=True,
    )


def _canonical_gate_layer(tensors: list[Array], parity: int, gate: Array) -> None:
    _right_canonicalize_open_mps(tensors)
    for site in range(parity):
        _move_open_mps_centre_right(tensors, site)
    bonds = list(range(parity, len(tensors) - 1, 2))
    for index, bond in enumerate(bonds):
        _apply_two_site_gate(tensors, bond, gate)
        if index + 1 < len(bonds):
            _move_open_mps_centre_right(tensors, bond + 1)
    _normalize_mps(tensors)


def _product_strang_open_mps(
    vector: Array,
    protocol: LocalEvolutionProtocol,
) -> list[Array]:
    tensors = _open_product_mps(vector, protocol.lightcone_sites)
    for parity, gate in (
        (0, protocol.half_gate),
        (1, protocol.full_gate),
        (0, protocol.half_gate),
    ):
        _canonical_gate_layer(tensors, parity, gate)
    return tensors


def _period_two_to_one_site(A_even: Array, A_odd: Array) -> Array:
    """Embed a two-site-periodic MPS pair into one off-diagonal tensor."""

    left_even, d, right_even = A_even.shape
    left_odd, d_odd, right_odd = A_odd.shape
    if d_odd != d or right_even != left_odd or right_odd != left_even:
        raise ValueError(
            "two-site tensors must have dimensions "
            "(D0,d,D1) and (D1,d,D0)"
        )
    tensor = np.zeros(
        (d, left_even + right_even, left_even + right_even),
        dtype=np.complex128,
    )
    for physical in range(d):
        tensor[physical, :left_even, left_even:] = A_even[:, physical, :]
        tensor[physical, left_even:, :left_even] = A_odd[:, physical, :]
    return tensor


def _stiefel_perturb(A: Array, amplitude: float, seed: int) -> Array:
    if amplitude <= 0.0:
        return np.asarray(A, dtype=np.complex128).copy()
    d, D, _ = A.shape
    rng = np.random.default_rng(seed)
    W = stack_tensor(A)
    noise = (rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)) / np.sqrt(
        2.0 * D
    )
    return unstack_tensor(polar_retraction(W + float(amplitude) * noise), d, D)


def _target_trace_distance(rho: Array, target: Array) -> float:
    difference = 0.5 * ((rho - target) + (rho - target).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


def product_circuit_left_canonical_seed(
    source: Array,
    target_bond_dimension: int,
    protocol: LocalEvolutionProtocol,
    *,
    mixing_amplitude: float = 1e-3,
    seed: int = 2,
    canonical_tolerance: float = 1e-10,
) -> tuple[Array, InitialLiftDiagnostics]:
    """Construct a product-start seed from the exact one-step circuit state.

    A second-order brickwall step applied to a product state is a shallow
    two-site-periodic MPS. Embedding the two central tensors into off-diagonal
    blocks gives a one-site left-canonical tensor with bond dimension
    ``D_even + D_odd``. A small deterministic Stiefel perturbation breaks the
    exact period-two transfer degeneracy; the first LOMPS update can then polish
    this seed back to the exact first target.
    """

    source = coerce_initial_tensor(source)
    d, source_D, _ = source.shape
    target_D = int(target_bond_dimension)
    if source_D != 1:
        raise ValueError("circuit product seed requires a product input tensor")
    if len(protocol.target_margins) != 1:
        raise ValueError("circuit product seed currently supports one target window")
    if protocol.lightcone_sites % 2:
        raise ValueError("circuit product seed requires an even light cone")
    if protocol.brickwall_unitary.shape != (d ** protocol.lightcone_sites,) * 2:
        raise ValueError("protocol unitary is incompatible with source dimension")

    source_errors = canonical_errors(source)
    if source_errors["left_canonical_error"] > canonical_tolerance:
        raise ValueError(
            "initial tensor is not left-canonical within tolerance; "
            "canonicalize it before passing it to LOMPS"
        )

    vector = source[:, 0, 0]
    tensors = _product_strang_open_mps(vector, protocol)
    left_margin, _ = protocol.target_margins[0]
    base = _period_two_to_one_site(tensors[left_margin], tensors[left_margin + 1])
    base_D = base.shape[1]
    if target_D != base_D:
        raise ValueError(
            "target_bond_dimension must match the exact product-circuit seed "
            f"dimension {base_D}; got {target_D}"
        )
    seed_tensor = _stiefel_perturb(base, mixing_amplitude, seed)

    seed_errors = canonical_errors(seed_tensor)
    seed_r, fixed = right_fixed_point(seed_tensor)
    target = protocol.target_rdm(source)
    seed_rho = block_rdm(seed_tensor, protocol.block_length, seed_r)
    target_frobenius = float(np.linalg.norm(seed_rho - target))
    target_trace_distance = _target_trace_distance(seed_rho, target)
    rdm_errors: dict[str, float] = {str(protocol.block_length): target_frobenius}
    for length in range(1, protocol.block_length):
        traced = tuple(range(length, protocol.block_length))
        reduced_target = partial_trace_sites(
            target,
            sites=protocol.block_length,
            traced_sites=traced,
        )
        rdm_errors[str(length)] = float(
            np.linalg.norm(block_rdm(seed_tensor, length, seed_r) - reduced_target)
        )

    diagnostics = InitialLiftDiagnostics(
        source_bond_dimension=source_D,
        target_bond_dimension=target_D,
        physical_dimension=d,
        noise_amplitude=float(mixing_amplitude),
        seed=int(seed),
        source_left_canonical_error=source_errors["left_canonical_error"],
        seed_left_canonical_error=seed_errors["left_canonical_error"],
        seed_transfer_gap=float(injectivity_diagnostics(seed_tensor)["gap"]),
        seed_right_fixed_point_minimum_eigenvalue=float(fixed["min_eigenvalue"]),
        seed_rdm_frobenius_error_by_length=rdm_errors,
        method="product_circuit",
        base_bond_dimension=base_D,
        seed_target_rdm_frobenius_error=target_frobenius,
        seed_target_trace_distance=target_trace_distance,
    )
    return seed_tensor, diagnostics


def lift_left_canonical_seed(
    source: Array,
    target_bond_dimension: int,
    *,
    noise_amplitude: float = 1e-8,
    seed: int = 104_729,
    diagnostic_block_length: int = 1,
    canonical_tolerance: float = 1e-10,
) -> tuple[Array, InitialLiftDiagnostics]:
    """Lift a low-D left-canonical source to a deterministic high-D seed.

    The returned seed is only an optimizer starting point. The physical first
    target should still be built from ``source`` when the source bond dimension
    is smaller than the trajectory bond dimension.

    The default noise amplitude follows the historical QDMT product-state
    start: embed into the upper-left virtual block, add tiny noise in the new
    virtual subspace, and QR-project back to the canonical manifold.
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
