#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def load(path):
    with Path(path).open("r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results")
    parser.add_argument("--defense", choices=["none", "q8", "q4", "q2", "all"], default="all")
    args = parser.parse_args()

    defenses = ["none", "q8", "q4", "q2"] if args.defense == "all" else [args.defense]
    print("RQ3 ConvNeXt-Tiny Tiny ImageNet 64x64 Split Learning")
    print("| Defense | Cut | Standalone Val Acc | SDAR FG Acc | SDAR Val Acc | MSE | PSNR | SSIM | LPIPS | ASR | Fixed SSIM |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for defense in defenses:
        for cut in ["c1", "c2", "c3", "c4"]:
            victim_path = Path(args.run_root) / defense / cut / "victim" / "victim_summary.json"
            sdar_path = Path(args.run_root) / defense / cut / "sdar" / "full_sdar_summary.json"
            if not victim_path.exists() or not sdar_path.exists():
                continue
            v = load(victim_path)
            s = load(sdar_path)
            lpips = s.get("lpips_mean")
            fixed_ssim = s.get("fixed_probe_ssim_mean")
            print(
                f"| {defense} | {cut.upper()} | {v.get('best_val_acc', -1):.4f} "
                f"| {s.get('fg_acc_last100', -1):.4f} "
                f"| {s.get('post_sdar_val_acc', -1):.4f} "
                f"| {s.get('mse_mean', -1):.6f} | {s.get('psnr_mean', -1):.2f} "
                f"| {s.get('ssim_mean', -1):.4f} "
                f"| {lpips if lpips is not None else -1:.6f} "
                f"| {s.get('asr_ssim_0.90', -1):.4f} "
                f"| {fixed_ssim if fixed_ssim is not None else -1:.4f} |"
            )


if __name__ == "__main__":
    main()
