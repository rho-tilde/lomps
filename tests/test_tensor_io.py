from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lomps.canonical import random_left_canonical
from lomps.convert_tensor import main as convert_main
from lomps.tensor_io import file_sha256, load_tensor_file


class TensorIOTests(unittest.TestCase):
    def test_native_npy_is_physical_first(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=901)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.npy"
            np.save(path, A)
            loaded, info = load_tensor_file(path)
        np.testing.assert_array_equal(loaded, A)
        self.assertEqual(info.detected_layout, "physical-left-right")
        self.assertIsNone(info.archive_key)

    def test_keyed_legacy_npz_is_converted(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=902)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(path, A=np.transpose(A, (1, 0, 2)))
            loaded, info = load_tensor_file(path, key="A")
        np.testing.assert_array_equal(loaded, A)
        self.assertEqual(info.archive_key, "A")
        self.assertEqual(info.detected_layout, "legacy-left-physical-right")

    def test_ambiguous_npz_requires_explicit_layout(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=903)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.npz"
            np.savez(path, A=A)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                load_tensor_file(path, key="A")
            loaded, _ = load_tensor_file(
                path,
                key="A",
                layout="physical-left-right",
            )
        np.testing.assert_array_equal(loaded, A)

    def test_converter_writes_native_tensor_and_hash_manifest(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=904)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "legacy.npz"
            output = directory / "converted.npy"
            np.savez(source, A=np.transpose(A, (1, 0, 2)))
            argv = [
                "lomps-convert-tensor",
                "--input",
                str(source),
                "--output",
                str(output),
                "--key",
                "A",
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                convert_main()
            manifest = json.loads(output.with_suffix(".json").read_text())
            converted = np.load(output)
            digest = file_sha256(output)
        np.testing.assert_array_equal(converted, A)
        self.assertEqual(manifest["output_sha256"], digest)
        self.assertLess(
            manifest["canonical_diagnostics"]["left_canonical_error"],
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
