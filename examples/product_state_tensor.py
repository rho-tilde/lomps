#!/usr/bin/env python
"""Create a qubit product-state vector accepted by ``lomps-run``.

The evolution runner accepts a one-site product vector with shape ``(2,)``.
This helper writes that vector as a small ``.npy`` file so launch commands are
fully reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def product_vector(name: str) -> np.ndarray:
    """Return one of the standard qubit product states."""

    normalized = {
        "x": np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0),
        "minus-x": np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0),
        "y": np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0),
        "minus-y": np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2.0),
        "z": np.array([1.0, 0.0], dtype=np.complex128),
        "minus-z": np.array([0.0, 1.0], dtype=np.complex128),
    }
    try:
        return normalized[name]
    except KeyError as error:
        raise ValueError(f"unknown product state {name!r}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        choices=("x", "minus-x", "y", "minus-y", "z", "minus-z"),
        default="y",
        help="Product state to write. The historical quench uses +y.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, product_vector(args.state))
    print(f"wrote {args.state} product vector to {args.output}")


if __name__ == "__main__":
    main()
