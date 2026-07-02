"""LOMPS: local optimization of matrix-product states."""

from .canonical import canonical_errors, random_left_canonical
from .optimizer import LMOptions, LMResult, optimize_tensor
from .protocol import LocalEvolutionProtocol, NONINTEGRABLE_ISING
from .rdm import block_rdm

__all__ = [
    "LMOptions",
    "LMResult",
    "LocalEvolutionProtocol",
    "NONINTEGRABLE_ISING",
    "block_rdm",
    "canonical_errors",
    "optimize_tensor",
    "random_left_canonical",
]

__version__ = "0.1.0"
