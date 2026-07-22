"""Convert a keyed or legacy-layout tensor into native LOMPS ``.npy`` form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .canonical import canonical_errors
from .tensor_io import TensorLayout, file_sha256, load_tensor_file


LAYOUT_CHOICES: tuple[TensorLayout, ...] = (
    "auto",
    "physical-left-right",
    "legacy-left-physical-right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", default=None)
    parser.add_argument("--source-layout", choices=LAYOUT_CHOICES, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".npy":
        raise ValueError("--output must end in .npy")
    manifest_path = args.output.with_suffix(".json")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        raise FileExistsError("output or provenance sidecar exists; use --overwrite")

    tensor, info = load_tensor_file(
        args.input,
        key=args.key,
        layout=args.source_layout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, tensor)
    manifest = {
        "format": "LOMPS tensor conversion v1",
        "tensor_convention": "(physical, left, right)",
        "input": info.to_json(),
        "output_path": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "dtype": str(tensor.dtype),
        "canonical_diagnostics": canonical_errors(tensor),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
