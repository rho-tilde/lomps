"""Local gates and finite-window circuit helpers used by LOMPS."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.linalg as la


Array = np.ndarray


def two_site_hamiltonian_tfim(
    *, g: float, h: float, J: float, symmetric_transverse: bool = False
) -> Array:
    """Return the two-site Ising Hamiltonian used by the quench protocol."""

    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sz = np.diag([1.0, -1.0]).astype(np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    transverse = np.kron(sx, identity)
    if symmetric_transverse:
        transverse = 0.5 * (transverse + np.kron(identity, sx))
    return J * np.kron(sz, sz) + g * transverse + h * np.kron(sz, identity)


def two_site_gate_tfim(
    *,
    g: float,
    h: float,
    J: float,
    dt: float,
    time_factor: float = 1.0,
    symmetric_transverse: bool = False,
) -> Array:
    """Construct ``exp(-i * time_factor * dt * H)``."""

    H = two_site_hamiltonian_tfim(
        g=g,
        h=h,
        J=J,
        symmetric_transverse=symmetric_transverse,
    )
    return la.expm(-1j * time_factor * dt * H)


def second_order_brickwall_unitary(
    U_half: Array,
    U_full: Array,
    *,
    sites: int,
) -> Array:
    """Build the even-half / odd-full / even-half Strang circuit."""

    U_half = np.asarray(U_half, dtype=np.complex128)
    U_full = np.asarray(U_full, dtype=np.complex128)
    local_dimension = round(np.sqrt(U_half.shape[0]))
    if sites % 2 or U_half.shape != U_full.shape:
        raise ValueError("requires an even site count and matching gates")
    identity = np.eye(local_dimension, dtype=np.complex128)

    def kron_all(factors: list[Array]) -> Array:
        result = np.array([[1.0 + 0.0j]])
        for factor in factors:
            result = np.kron(result, factor)
        return result

    even_layer = kron_all([U_half] * (sites // 2))
    odd_layer = kron_all(
        [identity] + [U_full] * (sites // 2 - 1) + [identity]
    )
    return even_layer @ odd_layer @ even_layer


def partial_trace_sites(
    rho: Array,
    *,
    sites: int,
    traced_sites: Iterable[int],
) -> Array:
    """Trace selected sites and normalize the remaining density matrix."""

    rho = np.asarray(rho, dtype=np.complex128)
    dimension = round(rho.shape[0] ** (1.0 / sites))
    if rho.shape != (dimension**sites, dimension**sites):
        raise ValueError("rho shape is incompatible with the site count")
    tensor = rho.reshape((dimension,) * (2 * sites))
    remaining = sites
    for site in sorted(set(int(x) for x in traced_sites), reverse=True):
        if site < 0 or site >= remaining:
            raise IndexError("traced site outside the current system")
        tensor = np.trace(tensor, axis1=site, axis2=site + remaining)
        remaining -= 1
    reduced = tensor.reshape(dimension**remaining, dimension**remaining)
    reduced = 0.5 * (reduced + reduced.conj().T)
    return reduced / np.trace(reduced)
