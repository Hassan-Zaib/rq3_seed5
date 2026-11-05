#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convnext_split import build_split_convnext_tiny, expected_activation_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    client, _ = build_split_convnext_tiny(args.cut)
    client.eval()
    x = torch.randn(args.batch_size, 3, 64, 64)
    with torch.no_grad():
        z = client(x)
    expected = (args.batch_size,) + expected_activation_shape(args.cut)
    print(f"{args.cut.upper()} activation shape: {tuple(z.shape)}")
    print(f"expected: {expected}")
    if tuple(z.shape) != expected:
        raise SystemExit(f"Shape mismatch for {args.cut}: got {tuple(z.shape)}, expected {expected}")


if __name__ == "__main__":
    main()

