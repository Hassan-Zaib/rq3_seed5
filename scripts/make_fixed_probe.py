#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_tinyimagenet import TinyImageNetDataset
from train_utils import save_image_grid, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--out-dir", default="results/fixed_probe")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9090)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TinyImageNetDataset(args.data_root, split="val", train=False)
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(ds), size=args.num_images, replace=False))

    xs, raws, labels, paths = [], [], [], []
    for idx in indices:
        x, label, raw, path, _ = ds[int(idx)]
        xs.append(x.numpy())
        raws.append(raw.numpy())
        labels.append(int(label))
        paths.append(path)

    np.save(out_dir / "fixed_probe_indices.npy", indices.astype(np.int64))
    np.save(out_dir / "fixed_probe_images_norm.npy", np.stack(xs, axis=0).astype(np.float32))
    np.save(out_dir / "fixed_probe_images.npy", np.stack(raws, axis=0).astype(np.float32))
    np.save(out_dir / "fixed_probe_labels.npy", np.asarray(labels, dtype=np.int64))
    save_json(
        out_dir / "fixed_probe_meta.json",
        {"seed": args.seed, "num_images": args.num_images, "indices": indices.tolist(), "paths": paths},
    )
    save_image_grid(out_dir / "fixed_probe_grid.png", [raws], ["Original"])
    print(f"Saved fixed probe to {out_dir}")


if __name__ == "__main__":
    main()

