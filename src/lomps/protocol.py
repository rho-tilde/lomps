"""Finite-window local evolution protocols."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import warnings

import numpy as np

from .gates import (
    partial_trace_sites,
    second_order_brickwall_density,
    second_order_brickwall_unitary,
    two_site_gate_tfim,
)
from .optimizer import FixedPointSolver, optimizer_right_fixed_point
from .rdm import block_rdm


Array = np.ndarray


@dataclass(frozen=True)
class LocalEvolutionProtocol:
    """Hamiltonian and finite light-cone definition for one local update."""

    name: str
    block_length: int
    delta_t: float
    g: float
    h: float
    J: float
    trotter_order: int = 2
    symmetric_transverse: bool = False
    odd_parity_warning_threshold: float = 1e-6
    target_contraction: str = "tensor"
    target_source_fixed_point_solver: FixedPointSolver = "dense"

    @property
    def lightcone_sites(self) -> int:
        left_margin, right_margin = self.target_margins[0]
        return self.block_length + left_margin + right_margin

    @property
    def target_margins(self) -> tuple[tuple[int, int], ...]:
        """Return left/right buffer sizes used to reduce the evolved light cone."""

        if self.trotter_order != 2:
            raise ValueError("LOMPS currently supports second-order updates")
        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        if self.block_length % 2 == 0:
            return ((2, 2),)
        return ((2, 3), (3, 2))

    @cached_property
    def half_gate(self) -> Array:
        return two_site_gate_tfim(
            g=self.g,
            h=self.h,
            J=self.J,
            dt=self.delta_t,
            time_factor=0.5,
            symmetric_transverse=self.symmetric_transverse,
        )

    @cached_property
    def full_gate(self) -> Array:
        return two_site_gate_tfim(
            g=self.g,
            h=self.h,
            J=self.J,
            dt=self.delta_t,
            time_factor=1.0,
            symmetric_transverse=self.symmetric_transverse,
        )

    @cached_property
    def brickwall_unitary(self) -> Array:
        return second_order_brickwall_unitary(
            self.half_gate,
            self.full_gate,
            sites=self.lightcone_sites,
        )

    def target_rdm_candidates(self, A: Array, r: Array | None = None) -> tuple[Array, ...]:
        """Return one or two reductions of the evolved light cone.

        Even block lengths use the symmetric ``2 | L | 2`` reduction. Odd block
        lengths use the two parity-related reductions ``2 | L | 3`` and
        ``3 | L | 2`` so that the Trotter circuit always acts on an even number
        of light-cone sites.
        """

        if self.target_contraction not in ("tensor", "dense"):
            raise ValueError("target_contraction must be 'tensor' or 'dense'")
        if self.target_source_fixed_point_solver not in ("dense", "fast"):
            raise ValueError(
                "target_source_fixed_point_solver must be 'dense' or 'fast'"
            )
        if r is None:
            r, _ = optimizer_right_fixed_point(A, self.target_source_fixed_point_solver)
        else:
            r = np.asarray(r, dtype=np.complex128)
        rho_large = block_rdm(A, self.lightcone_sites, r)
        if self.target_contraction == "dense":
            evolved = self.brickwall_unitary @ rho_large @ self.brickwall_unitary.conj().T
        else:
            evolved = second_order_brickwall_density(
                rho_large,
                self.half_gate,
                self.full_gate,
                sites=self.lightcone_sites,
            )
        candidates = []
        for left_margin, right_margin in self.target_margins:
            traced = tuple(range(left_margin)) + tuple(
                range(self.lightcone_sites - right_margin, self.lightcone_sites)
            )
            candidates.append(
                partial_trace_sites(
                    evolved,
                    sites=self.lightcone_sites,
                    traced_sites=traced,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _trace_distance(rho_a: Array, rho_b: Array) -> float:
        difference = 0.5 * ((rho_a - rho_b) + (rho_a - rho_b).conj().T)
        return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))

    def target_rdm(self, A: Array, r: Array | None = None) -> Array:
        """Evolve the light cone and retain the protocol's ``block_length`` sites."""

        candidates = self.target_rdm_candidates(A, r)
        if len(candidates) == 1:
            return candidates[0]

        trace_distance = self._trace_distance(candidates[0], candidates[1])
        hs_distance = float(np.linalg.norm(candidates[0] - candidates[1]))
        if trace_distance > self.odd_parity_warning_threshold:
            warnings.warn(
                "Odd-L parity targets differ: "
                f"trace_distance={trace_distance:.3e}, "
                f"hs_distance={hs_distance:.3e}",
                RuntimeWarning,
                stacklevel=2,
            )

        averaged = sum(candidates) / len(candidates)
        averaged = 0.5 * (averaged + averaged.conj().T)
        return averaged / np.trace(averaged)


NONINTEGRABLE_ISING = LocalEvolutionProtocol(
    name="nonintegrable_ising_L4",
    block_length=4,
    delta_t=1e-3,
    g=1.05,
    h=-0.5,
    J=-1.0,
    trotter_order=2,
    symmetric_transverse=False,
)


# Historical integrable quench convention:
#
#     H = - sum_j Z_j Z_{j+1} - g_physical sum_j X_j,
#     g_physical: 1.5 -> 0.2.
#
# The asymmetric bond Hamiltonian used by the original QDMT implementation is
# J ZZ + g (X \otimes I) + h (Z \otimes I). Consequently the post-quench
# physical field g_physical=+0.2 is represented below by the Python coefficient
# g=-0.2. Keeping the sign here, rather than repairing it at call sites, makes
# this preset reproduce the historical gate convention exactly.
INTEGRABLE_TFIM = LocalEvolutionProtocol(
    name="integrable_tfim_g1_0p2_L4",
    block_length=4,
    delta_t=1e-3,
    g=-0.2,
    h=0.0,
    J=-1.0,
    trotter_order=2,
    symmetric_transverse=False,
)


DEFAULT_PROTOCOL_PRESET = "nonintegrable-ising"
PROTOCOL_PRESETS: dict[str, LocalEvolutionProtocol] = {
    DEFAULT_PROTOCOL_PRESET: NONINTEGRABLE_ISING,
    "integrable-tfim": INTEGRABLE_TFIM,
}


def protocol_preset(name: str) -> LocalEvolutionProtocol:
    """Return a named immutable protocol preset."""

    try:
        return PROTOCOL_PRESETS[name]
    except KeyError as error:
        choices = ", ".join(sorted(PROTOCOL_PRESETS))
        raise ValueError(f"unknown protocol preset {name!r}; choose from {choices}") from error


def physical_product_state() -> Array:
    """Return the physical quench state ``(|0> + i|1>)/sqrt(2)``."""

    return np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
