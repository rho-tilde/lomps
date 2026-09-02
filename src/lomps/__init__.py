"""LOMPS: local optimization of matrix-product states."""

from .canonical import canonical_errors, random_left_canonical
from .dimensions import (
    minimum_bond_dimension_for_ti_rdm,
    quotient_tangent_dimension,
    translation_invariant_rdm_dimension,
)
from .embedding import (
    coerce_initial_tensor,
    lift_left_canonical_seed,
    product_circuit_left_canonical_seed,
    product_tensor,
)
from .fixed_target_cg import FixedTargetGrassmannCG, optimize_fixed_target_cg
from .integrable import exact_local_hs_rate, exact_transverse_magnetization
from .matrix_free_lm import (
    MatrixFreeLMOptions,
    MatrixFreeLMResult,
    optimize_fixed_target_matrix_free_lm,
)
from .optimizer import (
    CGOptions,
    LMOptions,
    LMResult,
    optimizer_right_fixed_point,
    optimize_tensor,
)
from .observables import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    local_expectations,
    one_site_rdm,
    trajectory_expectations,
)
from .protocol import (
    DEFAULT_PROTOCOL_PRESET,
    INTEGRABLE_TFIM,
    PROTOCOL_PRESETS,
    LocalEvolutionProtocol,
    NONINTEGRABLE_ISING,
    protocol_preset,
)
from .rdm import block_rdm
from .tensor_io import TensorLoadInfo, load_tensor_file

__all__ = [
    "LMOptions",
    "LMResult",
    "MatrixFreeLMOptions",
    "MatrixFreeLMResult",
    "CGOptions",
    "LocalEvolutionProtocol",
    "NONINTEGRABLE_ISING",
    "INTEGRABLE_TFIM",
    "DEFAULT_PROTOCOL_PRESET",
    "PROTOCOL_PRESETS",
    "protocol_preset",
    "TensorLoadInfo",
    "load_tensor_file",
    "block_rdm",
    "canonical_errors",
    "coerce_initial_tensor",
    "lift_left_canonical_seed",
    "minimum_bond_dimension_for_ti_rdm",
    "optimizer_right_fixed_point",
    "optimize_tensor",
    "optimize_fixed_target_matrix_free_lm",
    "local_expectations",
    "one_site_rdm",
    "trajectory_expectations",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "FixedTargetGrassmannCG",
    "optimize_fixed_target_cg",
    "product_circuit_left_canonical_seed",
    "product_tensor",
    "exact_local_hs_rate",
    "exact_transverse_magnetization",
    "quotient_tangent_dimension",
    "random_left_canonical",
    "translation_invariant_rdm_dimension",
]

__version__ = "0.1.0"
