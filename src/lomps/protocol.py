"""Finite-window local evolution protocols."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .gates import (
    partial_trace_sites,
    second_order_brickwall_unitary,
    two_site_gate_tfim,
)
from .rdm import block_rdm
from .transfer import right_fixed_point


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

    @property
    def lightcone_sites(self) -> int:
        if self.trotter_order != 2 or self.block_length % 2:
            raise ValueError("LOMPS currently supports even-L second-order updates")
        return self.block_length + 4

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

    def target_rdm(self, A: Array) -> Array:
        """Evolve the light cone and retain its central ``block_length`` sites."""

        r, _ = right_fixed_point(A)
        rho_large = block_rdm(A, self.lightcone_sites, r)
        evolved = self.brickwall_unitary @ rho_large @ self.brickwall_unitary.conj().T
        margin = (self.lightcone_sites - self.block_length) // 2
        traced = tuple(range(margin)) + tuple(
            range(self.lightcone_sites - margin, self.lightcone_sites)
        )
        return partial_trace_sites(
            evolved,
            sites=self.lightcone_sites,
            traced_sites=traced,
        )


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


def physical_product_state() -> Array:
    """Return the physical quench state ``(|0> + i|1>)/sqrt(2)``."""

    return np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
