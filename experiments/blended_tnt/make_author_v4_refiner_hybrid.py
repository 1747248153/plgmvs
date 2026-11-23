#!/usr/bin/env python3
"""Build an auditable author-backbone + v4-RefineAll checkpoint."""

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    author_ckpt = torch.load(args.author, map_location="cpu")
    refiner_ckpt = torch.load(args.refiner, map_location="cpu")
    author_state = author_ckpt["model"]
    refiner_state = refiner_ckpt["model"]

    common = set(author_state) & set(refiner_state)
    mismatched = [
        key for key in common
        if author_state[key].shape != refiner_state[key].shape
    ]
    if mismatched:
        raise RuntimeError(f"shape mismatch: {mismatched}")

    refiner_keys = sorted(
        key for key in refiner_state if key.startswith("depth_refiner.")
    )
    if len(refiner_keys) != 42:
        raise RuntimeError(f"expected 42 depth_refiner keys, got {len(refiner_keys)}")

    hybrid_state = dict(author_state)
    hybrid_state.update({key: refiner_state[key] for key in refiner_keys})
    if set(hybrid_state) != set(refiner_state):
        missing = sorted(set(refiner_state) - set(hybrid_state))
        extra = sorted(set(hybrid_state) - set(refiner_state))
        raise RuntimeError(f"hybrid key mismatch; missing={missing}, extra={extra}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": refiner_ckpt.get("epoch"),
        "model": hybrid_state,
        "hybrid_metadata": {
            "author_checkpoint": str(args.author.resolve()),
            "author_sha256": sha256(args.author),
            "refiner_checkpoint": str(args.refiner.resolve()),
            "refiner_sha256": sha256(args.refiner),
            "author_parameter_keys": len(author_state),
            "injected_refiner_keys": len(refiner_keys),
        },
    }, args.output)
    print(f"saved={args.output}")
    print(f"sha256={sha256(args.output)}")
    print(f"author_keys={len(author_state)} refiner_keys={len(refiner_keys)} total={len(hybrid_state)}")


if __name__ == "__main__":
    main()
