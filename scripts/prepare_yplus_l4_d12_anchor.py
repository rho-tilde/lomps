#!/usr/bin/env python3
"""Extract a reproducible adjacent tensor pair from the L=4,D=12 Y+ run."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from lomps.optimizer import optimizer_right_fixed_point
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm


Array = np.ndarray


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=root / "data" / "nonintegrable_d12_trajectory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def time_label(value: float) -> str:
    return f"t{value:.3f}".replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_cost(candidate: Array, target: Array) -> float:
    right, _ = optimizer_right_fixed_point(candidate, "dense")
    residual = block_rdm(candidate, 4, right) - target
    return 0.5 * float(np.vdot(residual, residual).real)


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    delta_t = 0.001
    physical_step = int(round(args.time / delta_t))
    if physical_step < 2 or abs(physical_step * delta_t - args.time) > 1e-12:
        raise ValueError("--time must be at least 0.002 and lie on the dt grid")

    trajectory_dir = args.trajectory_dir.resolve()
    try:
        source_trajectory = str(trajectory_dir.relative_to(repository_root))
    except ValueError:
        source_trajectory = str(trajectory_dir)
    states = np.load(trajectory_dir / "trajectory_states.npy", mmap_mode="r")
    times = np.load(trajectory_dir / "trajectory_times.npy", mmap_mode="r")

    def state_at(step: int) -> Array:
        if step < 2:
            raise ValueError("this extractor requires adjacent saved states")
        index = step - 2
        represented_time = step * delta_t
        if index >= len(times) or abs(float(times[index]) - represented_time) > 1e-12:
            raise RuntimeError(f"trajectory grid mismatch at t={represented_time}")
        return np.asarray(states[index], dtype=np.complex128)

    previous_time = args.time - delta_t
    previous = state_at(physical_step - 1)
    anchor = state_at(physical_step)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_name = f"previous_{time_label(previous_time)}_D12.npy"
    anchor_name = f"anchor_{time_label(args.time)}_D12.npy"
    previous_path = output / previous_name
    anchor_path = output / anchor_name
    np.save(previous_path, previous)
    np.save(anchor_path, anchor)

    protocol = replace(
        NONINTEGRABLE_ISING,
        block_length=4,
        target_source_fixed_point_solver="dense",
    )
    target = protocol.target_rdm(previous)
    source_metadata = json.loads(
        (trajectory_dir / "metadata.json").read_text()
    )
    manifest = {
        "schema_version": 1,
        "protocol": "nonintegrable_ising_yplus",
        "block_length": 4,
        "bond_dimension": 12,
        "delta_t": delta_t,
        "start_time": args.time,
        "source_trajectory": source_trajectory,
        "source_accuracy": {
            "accepted_cost_ceiling": source_metadata["optimizer"][
                "cost_tolerance"
            ],
            "optimizer_gradient_tolerance": source_metadata["optimizer"][
                "gradient_tolerance"
            ],
        },
        "files": {
            previous_name: {
                "physical_time": previous_time,
                "sha256": sha256(previous_path),
            },
            anchor_name: {
                "physical_time": args.time,
                "sha256": sha256(anchor_path),
            },
        },
        "anchor_input_exact_cost": exact_cost(anchor, target),
        "note": (
            "The anchor is refitted to the production internal target before "
            "the fixed-rho4 fibre search."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
