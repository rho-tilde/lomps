"""Dimension-count helpers for finite-window LOMPS fits."""

from __future__ import annotations

from math import isqrt


def _positive_integer(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def quotient_tangent_dimension(physical_dimension: int, bond_dimension: int) -> int:
    """Return the real uMPS quotient tangent dimension.

    For a one-site left-canonical tensor with physical dimension ``d`` and bond
    dimension ``D``, quotienting by the virtual gauge leaves
    ``2 * (d - 1) * D**2`` real physical tangent directions.
    """

    d = _positive_integer(physical_dimension, "physical_dimension")
    D = _positive_integer(bond_dimension, "bond_dimension")
    if d < 2:
        raise ValueError("physical_dimension must be at least 2")
    return 2 * (d - 1) * D * D


def translation_invariant_rdm_dimension(
    physical_dimension: int,
    block_length: int,
) -> int:
    """Return the real dimension of trace-fixed TI-compatible ``rho_L`` data.

    A generic trace-fixed density matrix on ``L`` sites has dimension
    ``d**(2L) - 1``. Translation invariance constrains the two ``(L - 1)``-site
    marginals to agree, reducing the relevant local data count to
    ``d**(2L) - d**(2L - 2)``.
    """

    d = _positive_integer(physical_dimension, "physical_dimension")
    L = _positive_integer(block_length, "block_length")
    if d < 2:
        raise ValueError("physical_dimension must be at least 2")
    return d ** (2 * L) - d ** (2 * (L - 1))


def minimum_bond_dimension_for_ti_rdm(
    physical_dimension: int,
    block_length: int,
) -> int:
    """Return the smallest ``D`` passing the TI local-RDM dimension count.

    This is a necessary dimension-count threshold for a generic local fit, not
    a global representability theorem. It solves
    ``2 * (d - 1) * D**2 >= d**(2L) - d**(2L - 2)``.
    """

    d = _positive_integer(physical_dimension, "physical_dimension")
    L = _positive_integer(block_length, "block_length")
    target = translation_invariant_rdm_dimension(d, L)
    denominator = 2 * (d - 1)
    required_square = (target + denominator - 1) // denominator
    root = isqrt(required_square)
    if root * root < required_square:
        root += 1
    return root
