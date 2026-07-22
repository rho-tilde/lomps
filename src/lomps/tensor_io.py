"""Unambiguous loading and provenance for LOMPS tensor inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np

from .embedding import coerce_initial_tensor


TensorLayout = Literal[
    "auto",
    "physical-left-right",
    "legacy-left-physical-right",
]


@dataclass(frozen=True)
class TensorLoadInfo:
    """Provenance recorded whenever a tensor file is interpreted."""

    source_path: str
    source_sha256: str
    archive_key: str | None
    requested_layout: str
    detected_layout: str
    source_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(path: Path, key: str | None) -> tuple[np.ndarray, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = tuple(archive.files)
            if key is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"{path} contains keys {keys}; select one with --initial-key"
                    )
                key = keys[0]
            if key not in archive:
                raise KeyError(f"key {key!r} is not present in {path}; available: {keys}")
            return np.asarray(archive[key]), key
    if key is not None:
        raise ValueError("an archive key can only be used with a .npz input")
    return np.asarray(np.load(path, allow_pickle=False)), None


def _convert_layout(
    value: np.ndarray,
    *,
    layout: TensorLayout,
    archive: bool,
) -> tuple[np.ndarray, str]:
    if value.ndim != 3:
        return value, "not-applicable"

    if layout == "physical-left-right":
        return value, layout
    if layout == "legacy-left-physical-right":
        return np.transpose(value, (1, 0, 2)), layout
    if layout != "auto":
        raise ValueError(f"unknown tensor layout {layout!r}")

    physical_first = value.shape[1] == value.shape[2]
    legacy = value.shape[0] == value.shape[2]
    if physical_first and not legacy:
        return value, "physical-left-right"
    if legacy and not physical_first:
        return np.transpose(value, (1, 0, 2)), "legacy-left-physical-right"
    if physical_first and legacy and not archive:
        # Native .npy inputs have always used the documented LOMPS convention.
        return value, "physical-left-right"
    if physical_first and legacy:
        raise ValueError(
            "the .npz tensor layout is ambiguous because all three axes have "
            "the same size; pass an explicit layout"
        )
    raise ValueError(
        f"cannot infer a tensor layout from shape {value.shape}; pass an explicit layout"
    )


def load_tensor_file(
    path: Path | str,
    *,
    key: str | None = None,
    layout: TensorLayout = "auto",
) -> tuple[np.ndarray, TensorLoadInfo]:
    """Load a LOMPS tensor and return its interpretation provenance.

    Native LOMPS arrays use ``(physical, left, right)``. Historical QDMT
    archives commonly use ``(left, physical, right)`` and may store the tensor
    under a named ``.npz`` key.
    """

    path = Path(path).expanduser().resolve()
    value, archive_key = _load_array(path, key)
    source_shape = tuple(int(x) for x in value.shape)
    converted, detected_layout = _convert_layout(
        value,
        layout=layout,
        archive=path.suffix.lower() == ".npz",
    )
    tensor = coerce_initial_tensor(converted)
    info = TensorLoadInfo(
        source_path=str(path),
        source_sha256=file_sha256(path),
        archive_key=archive_key,
        requested_layout=layout,
        detected_layout=detected_layout,
        source_shape=source_shape,
        output_shape=tuple(int(x) for x in tensor.shape),
    )
    return tensor, info
