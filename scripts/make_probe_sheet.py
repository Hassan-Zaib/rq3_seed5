#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_utils import save_image_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results")
    parser.add_argument(
        "--defense",
        required=True,
        choices=["none", "q8", "q4", "q2", "noise0.01", "noise0.05"],
    )
    parser.add_argument("--out-dir", default="results/probe_sheets")
    args = parser.parse_args()

    rows = []
    labels = []
    original = None
    for cut in ["c1", "c2", "c3", "c4"]:
        sdar_dir = Path(args.run_root) / args.defense / cut / "sdar"
        orig_path = sdar_dir / "fixed_probe_originals.npy"
        rec_path = sdar_dir / "fixed_probe_reconstructions.npy"
        if not rec_path.exists():
            raise FileNotFoundError(f"Missing fixed probe reconstruction: {rec_path}")
        if original is None:
            original = np.load(orig_path)
            rows.append(original)
            labels.append("Original")
        rows.append(np.load(rec_path))
        labels.append(cut.upper())

    out = Path(args.out_dir) / f"rq3_convnext_tiny_{args.defense}_fixed_probe_orig_c1_c4.png"
    save_image_grid(out, rows, labels)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
