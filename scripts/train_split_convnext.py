#!/usr/bin/env python
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convnext_split import build_split_convnext_tiny, expected_activation_shape
from data_tinyimagenet import build_loaders
from perturbations import apply_perturbation
from train_utils import AverageMeter, accuracy_top1, get_device, save_json, set_seed


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, target):
        return torch.mean(torch.sum(-target * torch.nn.functional.log_softmax(logits, dim=-1), dim=-1))


class ModelEmaPair:
    def __init__(self, client, server, decay=0.9999):
        self.client = copy.deepcopy(client).eval()
        self.server = copy.deepcopy(server).eval()
        self.decay = float(decay)
        for model in (self.client, self.server):
            for param in model.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def update(self, client, server):
        self._update_model(self.client, client)
        self._update_model(self.server, server)

    @torch.no_grad()
    def _update_model(self, ema_model, model):
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def smooth_one_hot(targets, num_classes, smoothing):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y = torch.full(
        (targets.size(0), num_classes),
        off_value,
        device=targets.device,
        dtype=torch.float32,
    )
    y.scatter_(1, targets.view(-1, 1), on_value)
    return y


def rand_bbox(width, height, lam, device):
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = torch.randint(width, (1,), device=device).item()
    cy = torch.randint(height, (1,), device=device).item()
    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))
    return x1, y1, x2, y2


def apply_mixup_cutmix(x, y, args, num_classes=200):
    target = smooth_one_hot(y, num_classes, args.label_smoothing)
    if args.mix_prob <= 0 or torch.rand((), device=x.device).item() > args.mix_prob:
        return x, target

    use_cutmix = args.cutmix_alpha > 0 and (
        args.mixup_alpha <= 0 or torch.rand((), device=x.device).item() < args.cutmix_switch_prob
    )
    alpha = args.cutmix_alpha if use_cutmix else args.mixup_alpha
    if alpha <= 0:
        return x, target

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_target = target * lam + target[perm] * (1.0 - lam)

    if use_cutmix:
        mixed_x = x.clone()
        _, _, height, width = x.shape
        x1, y1, x2, y2 = rand_bbox(width, height, lam, x.device)
        mixed_x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
        mixed_target = target * lam + target[perm] * (1.0 - lam)
        return mixed_x, mixed_target

    mixed_x = x * lam + x[perm] * (1.0 - lam)
    return mixed_x, mixed_target


def evaluate(client, server, loader, device, defense):
    client.eval()
    server.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y, _, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = apply_perturbation(client(x), defense)
            logits = server(z)
            loss = criterion(logits, y)
            loss_meter.update(loss.item(), x.size(0))
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return {"val_loss": loss_meter.avg, "val_acc": correct / max(1, total)}


def save_fixed_probe_outputs(client, server, device, run_dir, defense, fixed_probe_dir):
    fixed_probe_dir = Path(fixed_probe_dir)
    if not fixed_probe_dir.exists():
        return
    images_norm_path = fixed_probe_dir / "fixed_probe_images_norm.npy"
    labels_path = fixed_probe_dir / "fixed_probe_labels.npy"
    indices_path = fixed_probe_dir / "fixed_probe_indices.npy"
    if not images_norm_path.exists():
        return
    x = torch.from_numpy(np.load(images_norm_path)).to(device)
    client.eval()
    server.eval()
    with torch.no_grad():
        z_clean = client(x)
        z_def = apply_perturbation(z_clean, defense)
        logits = server(z_def)
    np.save(run_dir / "fixed_probe_smashed_clean.npy", z_clean.cpu().numpy().astype(np.float32))
    np.save(run_dir / "fixed_probe_smashed_defended.npy", z_def.cpu().numpy().astype(np.float32))
    np.save(run_dir / "fixed_probe_logits.npy", logits.cpu().numpy().astype(np.float32))
    if labels_path.exists():
        np.save(run_dir / "fixed_probe_labels.npy", np.load(labels_path))
    if indices_path.exists():
        np.save(run_dir / "fixed_probe_indices.npy", np.load(indices_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--fixed-probe-dir", default="results/fixed_probe")
    parser.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    parser.add_argument(
        "--defense",
        required=True,
        choices=["none", "q8", "q4", "q2", "noise0.01", "noise0.05"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-switch-prob", type=float, default=0.5)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--model-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-ema-decay", type=float, default=0.999)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    run_dir = Path(args.output_dir) / args.defense / args.cut / "victim"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args.data_root, args.batch_size, args.num_workers)
    client, server = build_split_convnext_tiny(
        args.cut, num_classes=200, drop_path_rate=args.drop_path_rate
    )
    client.to(device)
    server.to(device)

    with torch.no_grad():
        z_shape = tuple(client(torch.zeros(2, 3, 64, 64, device=device)).shape)
    expected = (2,) + expected_activation_shape(args.cut)
    print(f"{args.cut.upper()} activation shape: {z_shape}, expected {expected}", flush=True)
    if z_shape != expected:
        raise SystemExit(f"Shape mismatch: got {z_shape}, expected {expected}")

    criterion = SoftTargetCrossEntropy()
    optimizer = AdamW(
        list(client.parameters()) + list(server.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.warmup_epochs > 0 and args.epochs > args.warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = amp_enabled and args.amp_dtype == "fp16"
    scaler = GradScaler("cuda", enabled=use_scaler)
    ema = ModelEmaPair(client, server, decay=args.model_ema_decay) if args.model_ema else None
    history = []
    best_acc = -1.0
    started = time.time()

    config = vars(args).copy()
    config["activation_shape"] = z_shape
    config["expected_activation_shape"] = expected
    save_json(run_dir / "config.json", config)

    for epoch in range(1, args.epochs + 1):
        client.train()
        server.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y, _, _, _) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mixed_x, soft_y = apply_mixup_cutmix(x, y, args)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                z = client(mixed_x)
                z = apply_perturbation(z, args.defense)
                if not torch.isfinite(z).all():
                    raise FloatingPointError(
                        f"Non-finite smashed activation at epoch={epoch}, step={step}, "
                        f"cut={args.cut}, defense={args.defense}"
                    )
                logits = server(z)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Non-finite logits at epoch={epoch}, step={step}, "
                        f"cut={args.cut}, defense={args.defense}"
                    )
                loss = criterion(logits, soft_y)
                loss_for_backward = loss / args.update_freq
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}"
                )
            scaler.scale(loss_for_backward).backward()
            if step % args.update_freq == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(client.parameters()) + list(server.parameters()),
                    max_norm=args.grad_clip,
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch}, step={step}: "
                        f"{grad_norm.item()}"
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(client, server)
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(
                accuracy_top1(logits.detach(), soft_y.argmax(dim=1)),
                x.size(0),
            )

        scheduler.step()
        raw_val = evaluate(client, server, val_loader, device, args.defense)
        ema_val = None
        if ema is not None:
            ema_val = evaluate(ema.client, ema.server, val_loader, device, args.defense)
        if ema_val is not None and ema_val["val_acc"] >= raw_val["val_acc"]:
            eval_client = ema.client
            eval_server = ema.server
            val = ema_val
            val_source = "ema"
        else:
            eval_client = client
            eval_server = server
            val = raw_val
            val_source = "raw"
        row = {
            "epoch": epoch,
            "train_loss": loss_meter.avg,
            "train_acc": acc_meter.avg,
            "raw_val_loss": raw_val["val_loss"],
            "raw_val_acc": raw_val["val_acc"],
            "ema_val_loss": ema_val["val_loss"] if ema_val is not None else None,
            "ema_val_acc": ema_val["val_acc"] if ema_val is not None else None,
            "val_loss": val["val_loss"],
            "val_acc": val["val_acc"],
            "val_source": val_source,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        ema_text = f" ema_val_acc={row['ema_val_acc']:.4f}" if row["ema_val_acc"] is not None else ""
        print(
            f"[{args.defense} {args.cut}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"raw_val_acc={row['raw_val_acc']:.4f}{ema_text} "
            f"best_eval={row['val_acc']:.4f}({row['val_source']})",
            flush=True,
        )

        ckpt = {
            "client": eval_client.state_dict(),
            "server": eval_server.state_dict(),
            "client_raw": client.state_dict(),
            "server_raw": server.state_dict(),
            "epoch": epoch,
            "config": config,
            "history": history,
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if row["val_acc"] > best_acc:
            best_acc = row["val_acc"]
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        save_json(run_dir / "history.json", history)

    total_runtime = time.time() - started
    best_checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    eval_client, eval_server = build_split_convnext_tiny(
        args.cut, num_classes=200, drop_path_rate=args.drop_path_rate
    )
    eval_client.load_state_dict(best_checkpoint["client"])
    eval_server.load_state_dict(best_checkpoint["server"])
    eval_client.to(device).eval()
    eval_server.to(device).eval()
    save_fixed_probe_outputs(eval_client, eval_server, device, run_dir, args.defense, args.fixed_probe_dir)
    summary = {
        "cut": args.cut,
        "defense": args.defense,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "activation_shape": z_shape,
        "best_val_acc": best_acc,
        "final_val_acc": history[-1]["val_acc"],
        "final_raw_val_acc": history[-1]["raw_val_acc"],
        "final_ema_val_acc": history[-1]["ema_val_acc"],
        "final_val_source": history[-1]["val_source"],
        "final_train_acc": history[-1]["train_acc"],
        "train_acc_definition": "top1_against_dominant_mixup_cutmix_target",
        "total_runtime_sec": total_runtime,
        "history": history,
    }
    save_json(run_dir / "victim_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
