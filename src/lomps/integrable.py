"""Exact thermodynamic-limit diagnostics for the integrable TFIM quench."""

from __future__ import annotations

import numpy as np


def majorana_generator(sites: int, field: float) -> np.ndarray:
    """Return ``A`` for ``H=(i/4) w.T A w`` on a periodic TFIM ring."""

    if sites < 2:
        raise ValueError("sites must be at least two")
    generator = np.zeros((2 * sites, 2 * sites), dtype=float)
    for site in range(sites):
        first = 2 * site
        second = first + 1
        next_first = 2 * ((site + 1) % sites)
        generator[first, second] = 2.0 * field
        generator[second, first] = -2.0 * field
        generator[second, next_first] = -2.0
        generator[next_first, second] = 2.0
    return generator


def ground_state_covariance(generator: np.ndarray) -> np.ndarray:
    """Return the Majorana covariance of a quadratic ground state."""

    generator = np.asarray(generator, dtype=float)
    if (
        generator.ndim != 2
        or generator.shape[0] != generator.shape[1]
        or generator.shape[0] % 2
    ):
        raise ValueError("generator must be an even-dimensional square matrix")
    energies, modes = np.linalg.eigh(1j * generator)
    covariance = 1j * ((modes * np.sign(energies)) @ modes.conj().T)
    return np.real_if_close(covariance).real


def exact_local_hs_rate(
    times: float | np.ndarray,
    patch_size: int,
    *,
    g0: float = 1.5,
    g1: float = 0.2,
    ring_sites: int = 128,
) -> float | np.ndarray:
    """Return the exact local Hilbert--Schmidt Loschmidt rate.

    The result is ``-log(Tr[rho_L(t) rho_L(0)]) / L`` for a ``g0 -> g1``
    quench in ``H=-ZZ-gX``. The finite free-fermion ring must be chosen larger
    than the patch and the time-evolution light cone.
    """

    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if ring_sites < 4 * patch_size:
        raise ValueError("ring_sites must be at least four times patch_size")
    scalar = np.ndim(times) == 0
    flat_times = np.asarray(times, dtype=float).reshape(-1)

    initial_generator = majorana_generator(ring_sites, g0)
    final_generator = majorana_generator(ring_sites, g1)
    initial_covariance = ground_state_covariance(initial_generator)
    final_energies, final_modes = np.linalg.eigh(1j * final_generator)
    block_modes = final_modes[: 2 * patch_size]
    initial_block = initial_covariance[: 2 * patch_size, : 2 * patch_size]
    identity = np.eye(2 * patch_size)
    rates = np.empty(flat_times.size, dtype=float)

    for index, time in enumerate(flat_times):
        evolved_rows = (
            block_modes * np.exp(-1j * final_energies * float(time))
        ) @ final_modes.conj().T
        evolved_block = evolved_rows @ initial_covariance @ evolved_rows.T
        evolved_block = np.real_if_close(evolved_block).real
        sign, log_determinant = np.linalg.slogdet(
            identity - evolved_block @ initial_block
        )
        if sign <= 0:
            raise FloatingPointError("non-positive Gaussian overlap determinant")
        log_overlap = -patch_size * np.log(2.0) + 0.5 * log_determinant
        rates[index] = -log_overlap / patch_size
    if scalar:
        return float(rates[0])
    return rates.reshape(np.shape(times))


def exact_transverse_magnetization(
    times: float | np.ndarray,
    *,
    g0: float = 1.5,
    g1: float = 0.2,
    quadrature_points: int = 2048,
    chunk_size: int = 512,
) -> float | np.ndarray:
    """Return exact ``<X>(t)`` for ``g0 -> g1`` in ``H=-ZZ-gX``.

    The expression is the standard free-fermion momentum integral used by the
    historical QDMT analysis. Gauss-Legendre quadrature makes evaluation over
    an entire saved trajectory deterministic and inexpensive.
    """

    if quadrature_points < 16 or chunk_size < 1:
        raise ValueError("quadrature_points must be >=16 and chunk_size positive")
    scalar = np.ndim(times) == 0
    flat_times = np.asarray(times, dtype=float).reshape(-1)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    momenta = 0.5 * np.pi * (nodes + 1.0)
    weights = 0.5 * weights

    def theta(g: float) -> np.ndarray:
        return 0.5 * np.arctan2(np.sin(momenta), g - np.cos(momenta))

    theta0 = theta(g0)
    theta1 = theta(g1)
    delta = theta1 - theta0
    epsilon = 2.0 * np.sqrt(
        (g1 - np.cos(momenta)) ** 2 + np.sin(momenta) ** 2
    )
    stationary = np.cos(2.0 * theta1) * np.cos(2.0 * delta)
    oscillatory = np.sin(2.0 * theta1) * np.sin(2.0 * delta)

    result = np.empty_like(flat_times)
    for start in range(0, flat_times.size, chunk_size):
        stop = min(start + chunk_size, flat_times.size)
        phases = np.cos(2.0 * flat_times[start:stop, None] * epsilon[None, :])
        result[start:stop] = (stationary + phases * oscillatory) @ weights
    if scalar:
        return float(result[0])
    return result.reshape(np.shape(times))
