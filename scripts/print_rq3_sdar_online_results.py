#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def load(path):
    with Path(path).open("r") as f:
        return json.load(f)


def fmt(value, decimals=4):
    if value is None:
        return "-"
    return f"{float(value):.{decimals}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results_sdar_online")
    parser.add_argument(
        "--defense",
        choices=["none", "q8", "q4", "q2", "noise0.01", "noise0.05", "all"],
        default="all",
    )
    args = parser.parse_args()

    defenses = (
        ["none", "q8", "q4", "q2", "noise0.01", "noise0.05"]
        if args.defense == "all"
        else [args.defense]
    )
    print("RQ3 ConvNeXt-Tiny Tiny ImageNet 64x64: Clean Online SDAR")
    print("| Defense | Cut | Raw Val Acc | EMA Val Acc | FG Acc Last100 | MSE | PSNR | SSIM | LPIPS | ASR | Fixed SSIM | Steps | Batch | Attack Pool |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for defense in defenses:
        for cut in ["c1", "c2", "c3", "c4"]:
            path = Path(args.run_root) / defense / cut / "sdar" / "full_sdar_summary.json"
            if not path.exists():
                continue
            s = load(path)
            print(
                f"| {defense} | {cut.upper()} "
                f"| {fmt(s.get('post_sdar_val_acc'))} "
                f"| {fmt(s.get('post_sdar_ema_val_acc'))} "
                f"| {fmt(s.get('fg_acc_last100'))} "
                f"| {fmt(s.get('mse_mean'), 6)} "
                f"| {fmt(s.get('psnr_mean'), 2)} "
                f"| {fmt(s.get('ssim_mean'))} "
                f"| {fmt(s.get('lpips_mean'), 6)} "
                f"| {fmt(s.get('asr_ssim_0.90'))} "
                f"| {fmt(s.get('fixed_probe_ssim_mean'))} "
                f"| {int(s.get('attack_steps', -1))} "
                f"| {int(s.get('batch_size', -1))} "
                f"| {s.get('data_protocol', '-')} |"
            )


if __name__ == "__main__":
    main()
