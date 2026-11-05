#!/usr/bin/env python
import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convnext_split import build_split_convnext_tiny, expected_activation_shape
from data_tinyimagenet import TinyImageNetDataset, build_eval_loader, inverse_normalize
from full_sdar import (
    ConditionalDecoder,
    ConditionalFeatureDiscriminator,
    ConditionalImageDiscriminator,
    set_trainable,
)
from perturbations import apply_perturbation
from train_utils import (
    get_device,
    reconstruction_metric_summary,
    save_image_grid,
    save_json,
    set_seed,
)


class ModelEmaPair:
    def __init__(self, client, server, decay):
        self.client = copy.deepcopy(client).eval()
        self.server = copy.deepcopy(server).eval()
        self.decay = float(decay)
        for parameter in self.client.parameters():
            parameter.requires_grad_(False)
        for parameter in self.server.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, client, server):
        self._update_one(self.client, client)
        self._update_one(self.server, server)

    def _update_one(self, ema_model, model):
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


CKPT_KEEP = 2
# Arguments that may differ between the run that wrote a checkpoint and the run that resumes it.
CKPT_FREE_ARGS = {"ckpt_every", "ckpt_dir"}


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class ResumableLoader:
    """Draws batches exactly like next_batch() and records what is needed to rebuild the position after a
    restart: the epoch, the number of batches drawn in it, and the global torch RNG state when its iterator was
    created and when its first batch was drawn (the shuffle order and the worker seeds come from that RNG)."""

    def __init__(self, loader):
        self.loader = loader
        self.epoch = 0
        self._new_iterator()

    def _new_iterator(self):
        self.rng_at_iter = torch.get_rng_state()
        self.iterator = iter(self.loader)
        self.rng_at_first_batch = None
        self.batch_index = 0

    def _draw(self):
        if self.batch_index == 0:
            self.rng_at_first_batch = torch.get_rng_state()
        batch = next(self.iterator)
        self.batch_index += 1
        return batch

    def next(self):
        try:
            return self._draw()
        except StopIteration:
            self.epoch += 1
            self._new_iterator()
            return self._draw()

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "rng_at_iter": self.rng_at_iter,
            "rng_at_first_batch": self.rng_at_first_batch,
        }

    def load_state_dict(self, state):
        """Recreates the iterator and skips the batches already drawn. Changes the global torch RNG; the caller
        restores it afterwards."""
        self.epoch = state["epoch"]
        self.rng_at_iter = state["rng_at_iter"]
        torch.set_rng_state(self.rng_at_iter)
        self.iterator = iter(self.loader)
        self.rng_at_first_batch = state["rng_at_first_batch"]
        self.batch_index = 0
        if state["batch_index"] == 0:
            return
        torch.set_rng_state(self.rng_at_first_batch)
        # Without workers only the sampler is advanced (no images are loaded); with workers the batches are drawn.
        indices_only = self.loader.num_workers == 0 and hasattr(self.iterator, "_next_index")
        for _ in range(state["batch_index"]):
            if indices_only:
                self.iterator._next_index()
            else:
                next(self.iterator)
        self.batch_index = state["batch_index"]


def list_checkpoints(ckpt_dir):
    return sorted(Path(ckpt_dir).glob("ckpt_step*.pt"))


def save_checkpoint(ckpt_dir, step, payload):
    """Writes ckpt_step<step>.pt atomically (temp file, then os.replace) and keeps the last CKPT_KEEP."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_step{step:07d}.pt"
    tmp = ckpt_dir / f".{path.name}.tmp"
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    for old in list_checkpoints(ckpt_dir)[:-CKPT_KEEP]:
        old.unlink()
    return path


def load_latest_checkpoint(ckpt_dir):
    for path in reversed(list_checkpoints(ckpt_dir)):
        try:
            return path, torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"Cannot read checkpoint {path} ({exc}); trying the previous one", flush=True)
    return None, None


def autocast_context(device, enabled, amp_dtype):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def maybe_lpips(device, required, compute):
    if not compute:
        return None
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device).eval()
        return model
    except Exception as exc:
        if required:
            raise RuntimeError("LPIPS is required for the official RQ3 evaluation") from exc
        print(f"LPIPS unavailable, storing null LPIPS: {exc}", flush=True)
        return None


def build_victim(run_root, cut, defense, device, victim_init, drop_path_rate):
    client, server = build_split_convnext_tiny(
        cut, num_classes=200, drop_path_rate=drop_path_rate
    )
    checkpoint_path = None
    if victim_init == "checkpoint":
        checkpoint_path = Path(run_root) / defense / cut / "victim" / "checkpoint_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing victim checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        client.load_state_dict(checkpoint["client"])
        server.load_state_dict(checkpoint["server"])
    client.to(device).train()
    server.to(device).train()
    set_trainable(client, True)
    set_trainable(server, True)
    return client, server, checkpoint_path


def build_attack_loaders(data_root, batch_size, num_workers, seed, fixed_probe_dir, attack_pool):
    # Match the SDAR protocol: disjoint victim and auxiliary pools.
    # For the clean RQ3 online protocol we use train only, keeping Tiny ImageNet
    # validation held out for victim utility and reconstruction reporting.
    train_dataset = TinyImageNetDataset(data_root, split="train", train=False)
    if attack_pool == "train":
        full = train_dataset
        train_len = len(train_dataset)
        allow_probe_reassignment = False
    elif attack_pool == "trainval":
        val_dataset = TinyImageNetDataset(data_root, split="val", train=False)
        full = ConcatDataset([train_dataset, val_dataset])
        train_len = len(train_dataset)
        allow_probe_reassignment = True
    else:
        raise ValueError(f"Unsupported attack_pool={attack_pool}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full), generator=generator).tolist()
    middle = len(indices) // 2
    victim_indices = indices[:middle]
    auxiliary_indices = indices[middle:]

    probe_path = Path(fixed_probe_dir) / "fixed_probe_indices.npy"
    if allow_probe_reassignment and probe_path.exists():
        probe_indices = np.load(probe_path).astype(np.int64).tolist()
        probe_global = {train_len + int(index) for index in probe_indices}
        moved_to_victim = [index for index in auxiliary_indices if index in probe_global]
        if moved_to_victim:
            movable_victim = [index for index in victim_indices if index not in probe_global]
            swaps = movable_victim[: len(moved_to_victim)]
            victim_indices = [
                index for index in victim_indices if index not in set(swaps)
            ] + moved_to_victim
            auxiliary_indices = [
                index for index in auxiliary_indices if index not in probe_global
            ] + swaps
        if not probe_global.issubset(set(victim_indices)):
            raise RuntimeError("Fixed probes must be assigned exclusively to the victim pool")

    victim_subset = Subset(full, victim_indices)
    auxiliary_subset = Subset(full, auxiliary_indices)
    common = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return DataLoader(victim_subset, **common), DataLoader(auxiliary_subset, **common)


def build_attack_loaders_with_victim_transform(
    data_root, batch_size, num_workers, seed, fixed_probe_dir, attack_pool, victim_train_transform
):
    if not victim_train_transform:
        return build_attack_loaders(
            data_root, batch_size, num_workers, seed, fixed_probe_dir, attack_pool
        )
    if attack_pool != "train":
        raise ValueError("--victim-train-transform is only supported with --attack-pool train")
    victim_dataset = TinyImageNetDataset(data_root, split="train", train=True)
    auxiliary_dataset = TinyImageNetDataset(data_root, split="train", train=False)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(victim_dataset), generator=generator).tolist()
    middle = len(indices) // 2
    victim_indices = indices[:middle]
    auxiliary_indices = indices[middle:]
    common = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return (
        DataLoader(Subset(victim_dataset, victim_indices), **common),
        DataLoader(Subset(auxiliary_dataset, auxiliary_indices), **common),
    )


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def backward_and_step(loss, optimizer, parameters, name, grad_clip=0.0):
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite {name} loss: {loss.detach().item()}")
    loss.backward()
    parameters = list(parameters)
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient in {name}")
    if grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm in {name}")
    optimizer.step()


def build_fg_optimizer(parameters, args):
    parameters = list(parameters)
    if args.fg_optimizer == "adam":
        return Adam(parameters, lr=args.fg_lr)
    if args.fg_optimizer == "adamw":
        return AdamW(parameters, lr=args.fg_lr, weight_decay=args.fg_weight_decay)
    raise ValueError(f"Unsupported fg_optimizer={args.fg_optimizer}")


def set_fg_lr(optimizer, step, total_steps, args):
    if args.fg_schedule == "constant":
        lr = args.fg_lr
    elif args.fg_schedule == "warmup_cosine":
        warmup = max(0, int(args.fg_warmup_steps))
        if warmup > 0 and step <= warmup:
            lr = args.fg_lr * step / warmup
        else:
            decay_steps = max(1, total_steps - warmup)
            progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = args.fg_min_lr + (args.fg_lr - args.fg_min_lr) * cosine
    else:
        raise ValueError(f"Unsupported fg_schedule={args.fg_schedule}")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, target):
        return torch.mean(
            torch.sum(-target * torch.nn.functional.log_softmax(logits, dim=-1), dim=-1)
        )


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


def apply_victim_mix(x, y, args):
    target = smooth_one_hot(y, 200, args.fg_label_smoothing)
    if args.fg_mix_prob <= 0 or torch.rand((), device=x.device).item() > args.fg_mix_prob:
        return x, target, y, inverse_normalize(x)

    use_cutmix = args.fg_cutmix_alpha > 0 and (
        args.fg_mixup_alpha <= 0
        or torch.rand((), device=x.device).item() < args.fg_cutmix_switch_prob
    )
    alpha = args.fg_cutmix_alpha if use_cutmix else args.fg_mixup_alpha
    if alpha <= 0:
        return x, target, y, inverse_normalize(x)

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_target = target * lam + target[perm] * (1.0 - lam)
    dominant_labels = y if lam >= 0.5 else y[perm]

    if use_cutmix:
        mixed_x = x.clone()
        _, _, height, width = x.shape
        x1, y1, x2, y2 = rand_bbox(width, height, lam, x.device)
        mixed_x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
        mixed_target = target * lam + target[perm] * (1.0 - lam)
        dominant_labels = y if lam >= 0.5 else y[perm]
        return mixed_x, mixed_target, dominant_labels, inverse_normalize(mixed_x)

    mixed_x = x * lam + x[perm] * (1.0 - lam)
    return mixed_x, mixed_target, dominant_labels, inverse_normalize(mixed_x)


def eval_decoder(decoder, client, loader, device, defense, max_samples):
    decoder.eval()
    originals, reconstructions, labels = [], [], []
    with torch.no_grad():
        for x, y, raw, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)
            z = apply_perturbation(client(x), defense)
            reconstruction = decoder(z, y_device).cpu()
            originals.append(raw)
            reconstructions.append(reconstruction)
            labels.append(y)
            if sum(item.size(0) for item in originals) >= max_samples:
                break
    return (
        torch.cat(originals)[:max_samples].numpy(),
        torch.cat(reconstructions)[:max_samples].numpy(),
        torch.cat(labels)[:max_samples].numpy(),
    )


def reconstruct_fixed_probe(decoder, client, device, defense, fixed_probe_dir, out_dir):
    fixed_probe_dir = Path(fixed_probe_dir)
    norm_path = fixed_probe_dir / "fixed_probe_images_norm.npy"
    raw_path = fixed_probe_dir / "fixed_probe_images.npy"
    labels_path = fixed_probe_dir / "fixed_probe_labels.npy"
    indices_path = fixed_probe_dir / "fixed_probe_indices.npy"
    if not norm_path.exists() or not labels_path.exists():
        return None
    x = torch.from_numpy(np.load(norm_path)).to(device)
    labels = torch.from_numpy(np.load(labels_path)).long().to(device)
    originals = np.load(raw_path)
    with torch.no_grad():
        z = apply_perturbation(client(x), defense)
        reconstructions = decoder(z, labels).cpu().numpy()
    np.save(out_dir / "fixed_probe_originals.npy", originals.astype(np.float32))
    np.save(out_dir / "fixed_probe_reconstructions.npy", reconstructions.astype(np.float32))
    np.save(out_dir / "fixed_probe_labels.npy", labels.cpu().numpy())
    if indices_path.exists():
        np.save(out_dir / "fixed_probe_indices.npy", np.load(indices_path))
    save_image_grid(
        out_dir / "fixed_probe_grid.png",
        [originals, reconstructions],
        ["Original", "Full SDAR"],
    )
    return originals, reconstructions


def evaluate_victim(client, server, loader, device, defense):
    client.eval()
    server.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = server(apply_perturbation(client(x), defense))
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--run-root", default="results")
    parser.add_argument("--fixed-probe-dir", default="fixed_probe")
    parser.add_argument("--cut", required=True, choices=["c1", "c2", "c3", "c4"])
    parser.add_argument(
        "--defense",
        required=True,
        choices=["none", "q8", "q4", "q2", "noise0.01", "noise0.05"],
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--attack-pool",
        choices=["train", "trainval"],
        default="train",
        help="train keeps validation held out; trainval reproduces the older combined-pool runner.",
    )
    parser.add_argument("--e-lr", type=float, default=1e-3)
    parser.add_argument("--d-lr", type=float, default=5e-4)
    parser.add_argument("--e-dis-lr", type=float, default=2e-5)
    parser.add_argument("--d-dis-lr", type=float, default=5e-9)
    parser.add_argument("--lambda1", type=float, default=0.02)
    parser.add_argument("--lambda2", type=float, default=1e-5)
    parser.add_argument(
        "--victim-init",
        choices=["scratch", "checkpoint"],
        default="scratch",
        help="scratch matches official SDAR; checkpoint is a warm-start ablation.",
    )
    parser.add_argument("--fg-lr", type=float, default=1e-3)
    parser.add_argument(
        "--fg-optimizer",
        choices=["adam", "adamw"],
        default="adam",
        help="Victim f/g optimizer. adam preserves the SDAR baseline; adamw is the ConvNeXt repair.",
    )
    parser.add_argument("--fg-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--fg-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
        help="Victim f/g learning-rate schedule.",
    )
    parser.add_argument("--fg-warmup-steps", type=int, default=0)
    parser.add_argument("--fg-min-lr", type=float, default=0.0)
    parser.add_argument("--fg-label-smoothing", type=float, default=0.0)
    parser.add_argument("--fg-mix-prob", type=float, default=0.0)
    parser.add_argument("--fg-mixup-alpha", type=float, default=0.8)
    parser.add_argument("--fg-cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--fg-cutmix-switch-prob", type=float, default=0.5)
    parser.add_argument(
        "--victim-train-transform",
        action="store_true",
        help="Use training transforms for victim batches in online SDAR.",
    )
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--model-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-ema-decay", type=float, default=0.9999)
    parser.add_argument(
        "--require-lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compute-lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fg-grad-clip",
        type=float,
        default=0.0,
        help="0 matches official SDAR; set a positive value only as a documented stability adaptation.",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=2000,
        help="save a resumable checkpoint every N steps (0: only on SIGTERM/SIGUSR1)",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="checkpoint folder; default <run-root>/<defense>/<cut>/sdar/checkpoints",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.run_root) / args.defense / args.cut / "sdar"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / "checkpoints"

    victim_client, victim_server, checkpoint_path = build_victim(
        args.run_root,
        args.cut,
        args.defense,
        device,
        args.victim_init,
        args.drop_path_rate,
    )
    simulator, _ = build_split_convnext_tiny(
        args.cut, num_classes=200, drop_path_rate=args.drop_path_rate
    )
    simulator = simulator.to(device)
    channels, height, _ = expected_activation_shape(args.cut)
    feature_discriminator = ConditionalFeatureDiscriminator(
        channels, height, num_classes=200
    ).to(device)
    decoder = ConditionalDecoder(channels, height, num_classes=200, out_hw=64).to(device)
    image_discriminator = ConditionalImageDiscriminator(num_classes=200, image_hw=64).to(device)

    e_optimizer = Adam(simulator.parameters(), lr=args.e_lr)
    e_dis_optimizer = Adam(feature_discriminator.parameters(), lr=args.e_dis_lr)
    d_optimizer = Adam(decoder.parameters(), lr=args.d_lr)
    d_dis_optimizer = Adam(image_discriminator.parameters(), lr=args.d_dis_lr)
    fg_parameters = list(victim_client.parameters()) + list(victim_server.parameters())
    fg_optimizer = build_fg_optimizer(fg_parameters, args)
    ema = ModelEmaPair(victim_client, victim_server, args.model_ema_decay) if args.model_ema else None
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=args.fg_label_smoothing)
    soft_cross_entropy = SoftTargetCrossEntropy()
    adversarial_loss = nn.BCEWithLogitsLoss()

    victim_loader, auxiliary_loader = build_attack_loaders_with_victim_transform(
        args.data_root,
        args.batch_size,
        args.num_workers,
        args.split_seed,
        args.fixed_probe_dir,
        args.attack_pool,
        args.victim_train_transform,
    )
    victim_stream = ResumableLoader(victim_loader)
    auxiliary_stream = ResumableLoader(auxiliary_loader)
    history = {
        key: []
        for key in (
            "eg_loss",
            "eg_acc",
            "e_gen_loss",
            "e_loss",
            "e_dis_real_loss",
            "e_dis_fake_loss",
            "e_dis_real_acc",
            "e_dis_fake_acc",
            "e_dis_loss",
            "d_mse_loss",
            "d_gen_loss",
            "d_loss",
            "d_dis_real_loss",
            "d_dis_fake_loss",
            "d_dis_real_acc",
            "d_dis_fake_acc",
            "d_dis_loss",
            "attack_loss",
            "fg_loss",
            "fg_acc",
            "fg_lr",
        )
    }
    started = time.time()

    ckpt_models = {
        "victim_client": victim_client,
        "victim_server": victim_server,
        "simulator": simulator,
        "feature_discriminator": feature_discriminator,
        "decoder": decoder,
        "image_discriminator": image_discriminator,
    }
    if ema is not None:
        ckpt_models["ema_victim_client"] = ema.client
        ckpt_models["ema_victim_server"] = ema.server
    ckpt_optimizers = {
        "fg": fg_optimizer,
        "e": e_optimizer,
        "e_dis": e_dis_optimizer,
        "d": d_optimizer,
        "d_dis": d_dis_optimizer,
    }
    ckpt_args = {key: value for key, value in vars(args).items() if key not in CKPT_FREE_ARGS}

    def checkpoint_payload(step):
        # No LR scheduler objects (set_fg_lr derives the LR from the step) and no GradScaler (bf16 autocast).
        return {
            "step": step,
            "args": ckpt_args,
            "models": {name: model.state_dict() for name, model in ckpt_models.items()},
            "optimizers": {name: opt.state_dict() for name, opt in ckpt_optimizers.items()},
            "loaders": {"victim": victim_stream.state_dict(), "auxiliary": auxiliary_stream.state_dict()},
            "rng": rng_state(),
            "history": history,
            "elapsed_sec": time.time() - started,
        }

    start_step = 1
    ckpt_path, ckpt = load_latest_checkpoint(ckpt_dir)
    if ckpt is not None:
        changed = sorted(
            key for key in set(ckpt["args"]) | set(ckpt_args) if ckpt["args"].get(key) != ckpt_args.get(key)
        )
        if changed:
            raise RuntimeError(f"Checkpoint {ckpt_path} was written with different arguments: {changed}")
        for name, model in ckpt_models.items():
            model.load_state_dict(ckpt["models"][name])
        for name, opt in ckpt_optimizers.items():
            opt.load_state_dict(ckpt["optimizers"][name])
        victim_stream.load_state_dict(ckpt["loaders"]["victim"])
        auxiliary_stream.load_state_dict(ckpt["loaders"]["auxiliary"])
        history = ckpt["history"]
        started = time.time() - ckpt["elapsed_sec"]
        set_rng_state(ckpt["rng"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {ckpt_path}: continuing at step {start_step}/{args.steps}", flush=True)
        del ckpt

    # On SIGTERM/SIGUSR1 (e.g. a cluster time limit) finish the current step, save a checkpoint and exit.
    stop_signal = []
    ckpt_signals = [signal.SIGTERM] + ([signal.SIGUSR1] if hasattr(signal, "SIGUSR1") else [])
    previous_handlers = {
        signum: signal.signal(signum, lambda signum, frame: stop_signal.append(signum)) for signum in ckpt_signals
    }

    for step in range(start_step, args.steps + 1):
        victim_batch = victim_stream.next()
        auxiliary_batch = auxiliary_stream.next()
        x, y, raw_x = victim_batch[0], victim_batch[1], victim_batch[2]
        xs, ys, raw_xs = auxiliary_batch[0], auxiliary_batch[1], auxiliary_batch[2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        raw_x = raw_x.to(device, non_blocking=True)
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        raw_xs = raw_xs.to(device, non_blocking=True)

        victim_client.train()
        victim_server.train()
        fg_lr = set_fg_lr(fg_optimizer, step, args.steps, args)
        fg_optimizer.zero_grad(set_to_none=True)
        victim_input, victim_target, attack_labels, attack_target = apply_victim_mix(x, y, args)
        with autocast_context(device, args.amp, args.amp_dtype):
            z_real = apply_perturbation(victim_client(victim_input), args.defense)
            if not torch.isfinite(z_real).all():
                raise FloatingPointError(f"Non-finite victim activation at SDAR step {step}")
            victim_logits = victim_server(z_real)
            if args.fg_mix_prob > 0:
                fg_loss = soft_cross_entropy(victim_logits, victim_target)
            else:
                fg_loss = cross_entropy(victim_logits, y)
        backward_and_step(
            fg_loss,
            fg_optimizer,
            fg_parameters,
            "fg",
            args.fg_grad_clip,
        )
        if ema is not None:
            ema.update(victim_client, victim_server)
        fg_acc = (victim_logits.argmax(dim=1) == victim_target.argmax(dim=1)).float().mean()
        z_real = z_real.detach()
        attack_labels = attack_labels.detach()
        attack_target = attack_target.detach()

        # Simulator: task consistency through the victim server plus feature adversary.
        set_trainable(victim_server, False)
        victim_server.eval()
        set_trainable(feature_discriminator, False)
        e_optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp, args.amp_dtype):
            z_simulated = apply_perturbation(simulator(xs), args.defense)
            if not torch.isfinite(z_simulated).all():
                raise FloatingPointError(f"Non-finite simulated activation at SDAR step {step}")
            simulated_logits = victim_server(z_simulated)
            eg_loss = cross_entropy(simulated_logits, ys)
            e_gen_loss = adversarial_loss(
                feature_discriminator(z_simulated, ys),
                torch.ones(args.batch_size, device=device),
            )
            e_loss = eg_loss + args.lambda1 * e_gen_loss
        backward_and_step(e_loss, e_optimizer, simulator.parameters(), "simulator")
        set_trainable(feature_discriminator, True)
        set_trainable(victim_server, True)

        # Feature discriminator: observed defended activations versus simulated ones.
        e_dis_optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp, args.amp_dtype):
            e_dis_real_logits = feature_discriminator(z_real.detach(), attack_labels)
            e_dis_fake_logits = feature_discriminator(z_simulated.detach(), ys)
            e_dis_real = adversarial_loss(
                e_dis_real_logits,
                torch.ones(args.batch_size, device=device),
            )
            e_dis_fake = adversarial_loss(
                e_dis_fake_logits,
                torch.zeros(args.batch_size, device=device),
            )
            e_dis_loss = e_dis_real + e_dis_fake
        backward_and_step(
            e_dis_loss,
            e_dis_optimizer,
            feature_discriminator.parameters(),
            "feature_discriminator",
        )

        # Decoder: auxiliary reconstruction plus image-space adversarial regularization.
        set_trainable(image_discriminator, False)
        d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with autocast_context(device, args.amp, args.amp_dtype):
                z_simulated_updated = apply_perturbation(simulator(xs), args.defense)
        with autocast_context(device, args.amp, args.amp_dtype):
            decoded_auxiliary = decoder(z_simulated_updated.detach(), ys)
            d_mse_loss = torch.mean((raw_xs - decoded_auxiliary.float()) ** 2)
            decoded_victim = decoder(z_real.detach(), attack_labels)
            d_gen_loss = adversarial_loss(
                image_discriminator(decoded_victim, attack_labels),
                torch.ones(args.batch_size, device=device),
            )
            d_loss = d_mse_loss + args.lambda2 * d_gen_loss
        backward_and_step(d_loss, d_optimizer, decoder.parameters(), "decoder")
        set_trainable(image_discriminator, True)

        # Image discriminator: real auxiliary images versus victim reconstructions.
        d_dis_optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp, args.amp_dtype):
            d_dis_real_logits = image_discriminator(raw_xs, ys)
            d_dis_fake_logits = image_discriminator(decoded_victim.detach(), attack_labels)
            d_dis_real = adversarial_loss(
                d_dis_real_logits,
                torch.ones(args.batch_size, device=device),
            )
            d_dis_fake = adversarial_loss(
                d_dis_fake_logits,
                torch.zeros(args.batch_size, device=device),
            )
            d_dis_loss = d_dis_real + d_dis_fake
        backward_and_step(
            d_dis_loss,
            d_dis_optimizer,
            image_discriminator.parameters(),
            "image_discriminator",
        )

        with torch.no_grad():
            with autocast_context(device, args.amp, args.amp_dtype):
                decoded_for_attack = decoder(z_real, attack_labels)
            attack_loss = torch.mean((attack_target - decoded_for_attack.float()) ** 2)
            eg_acc = (simulated_logits.argmax(dim=1) == ys).float().mean()
            e_dis_real_acc = (e_dis_real_logits >= 0).float().mean()
            e_dis_fake_acc = (e_dis_fake_logits < 0).float().mean()
            d_dis_real_acc = (d_dis_real_logits >= 0).float().mean()
            d_dis_fake_acc = (d_dis_fake_logits < 0).float().mean()

        values = {
            "eg_loss": eg_loss,
            "eg_acc": eg_acc,
            "e_gen_loss": e_gen_loss,
            "e_loss": e_loss,
            "e_dis_real_loss": e_dis_real,
            "e_dis_fake_loss": e_dis_fake,
            "e_dis_real_acc": e_dis_real_acc,
            "e_dis_fake_acc": e_dis_fake_acc,
            "e_dis_loss": e_dis_loss,
            "d_mse_loss": d_mse_loss,
            "d_gen_loss": d_gen_loss,
            "d_loss": d_loss,
            "d_dis_real_loss": d_dis_real,
            "d_dis_fake_loss": d_dis_fake,
            "d_dis_real_acc": d_dis_real_acc,
            "d_dis_fake_acc": d_dis_fake_acc,
            "d_dis_loss": d_dis_loss,
            "attack_loss": attack_loss,
            "fg_loss": fg_loss,
            "fg_acc": fg_acc,
            "fg_lr": torch.tensor(fg_lr, device=device),
        }
        for key, value in values.items():
            scalar = float(value.detach().item())
            if not np.isfinite(scalar):
                raise FloatingPointError(f"Non-finite {key} at SDAR step {step}: {scalar}")
            history[key].append(scalar)

        if step == 1 or step % 100 == 0:
            print(
                f"[full SDAR {args.defense} {args.cut}] {step}/{args.steps} "
                f"e={np.mean(history['e_loss'][-100:]):.4f} "
                f"e_dis={np.mean(history['e_dis_loss'][-100:]):.4f} "
                f"d={np.mean(history['d_loss'][-100:]):.4f} "
                f"d_dis={np.mean(history['d_dis_loss'][-100:]):.4f} "
                f"attack={np.mean(history['attack_loss'][-100:]):.4f} "
                f"fg_lr={fg_lr:.6g}",
                flush=True,
            )

        if stop_signal or (args.ckpt_every > 0 and step % args.ckpt_every == 0):
            path = save_checkpoint(ckpt_dir, step, checkpoint_payload(step))
            if stop_signal:
                print(f"Signal {stop_signal[0]}: saved {path} at step {step}; rerun the same command to resume", flush=True)
                sys.exit(128 + stop_signal[0])

    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)

    validation_loader = build_eval_loader(
        args.data_root, split="val", batch_size=args.batch_size, num_workers=args.num_workers
    )
    post_sdar_val_acc = evaluate_victim(
        victim_client, victim_server, validation_loader, device, args.defense
    )
    post_sdar_ema_val_acc = None
    if ema is not None:
        post_sdar_ema_val_acc = evaluate_victim(
            ema.client, ema.server, validation_loader, device, args.defense
        )
    originals, reconstructions, labels = eval_decoder(
        decoder,
        victim_client,
        validation_loader,
        device,
        args.defense,
        max_samples=args.eval_samples,
    )
    np.save(out_dir / "originals.npy", originals.astype(np.float32))
    np.save(out_dir / "reconstructions.npy", reconstructions.astype(np.float32))
    np.save(out_dir / "labels.npy", labels.astype(np.int64))
    np.save(out_dir / "full_sdar_history.npy", history, allow_pickle=True)
    save_image_grid(
        out_dir / "original_recon_grid.png",
        [originals[:20], reconstructions[:20]],
        ["Original", "Full SDAR"],
    )

    fixed = reconstruct_fixed_probe(
        decoder,
        victim_client,
        device,
        args.defense,
        args.fixed_probe_dir,
        out_dir,
    )
    lpips_model = maybe_lpips(device, args.require_lpips, args.compute_lpips)
    metrics = reconstruction_metric_summary(
        originals, reconstructions, lpips_model=lpips_model, device=device
    )
    if fixed is not None:
        fixed_metrics = reconstruction_metric_summary(
            fixed[0], fixed[1], lpips_model=lpips_model, device=device
        )
        metrics.update({f"fixed_probe_{key}": value for key, value in fixed_metrics.items()})

    config = {
        "e_lr": args.e_lr,
        "d_lr": args.d_lr,
        "e_dis_lr": args.e_dis_lr,
        "d_dis_lr": args.d_dis_lr,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
    }
    summary = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "attack": "full_sdar",
        "conditional": True,
        "victim_training": "online_joint_with_attack",
        "victim_init": args.victim_init,
        "victim_frozen": False,
        "victim_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "observed_interface": "post_perturbation_z_tilde",
        "cut": args.cut,
        "defense": args.defense,
        "attack_steps": args.steps,
        "batch_size": args.batch_size,
        "data_protocol": f"online_sdar_disjoint_50_50_{args.attack_pool}_victim_auxiliary_pool",
        "fixed_probe_pool": "heldout_val" if args.attack_pool == "train" else "victim_private_only",
        "config": config,
        "flip_rate": 0.0,
        "fg_lr": args.fg_lr,
        "fg_optimizer": args.fg_optimizer,
        "fg_weight_decay": args.fg_weight_decay,
        "fg_schedule": args.fg_schedule,
        "fg_warmup_steps": args.fg_warmup_steps,
        "fg_min_lr": args.fg_min_lr,
        "fg_label_smoothing": args.fg_label_smoothing,
        "fg_mix_prob": args.fg_mix_prob,
        "fg_mixup_alpha": args.fg_mixup_alpha,
        "fg_cutmix_alpha": args.fg_cutmix_alpha,
        "fg_cutmix_switch_prob": args.fg_cutmix_switch_prob,
        "victim_train_transform": args.victim_train_transform,
        "fg_grad_clip": args.fg_grad_clip,
        "drop_path_rate": args.drop_path_rate,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype if args.amp else None,
        "model_ema": args.model_ema,
        "model_ema_decay": args.model_ema_decay if args.model_ema else None,
        "post_sdar_val_acc": post_sdar_val_acc,
        "post_sdar_ema_val_acc": post_sdar_ema_val_acc,
        "total_runtime_sec": time.time() - started,
        **{f"{key}_last100": float(np.mean(value[-100:])) for key, value in history.items()},
        **metrics,
    }
    save_json(out_dir / "full_sdar_summary.json", summary)
    torch.save(
        {
            "victim_client": victim_client.state_dict(),
            "victim_server": victim_server.state_dict(),
            "simulator": simulator.state_dict(),
            "feature_discriminator": feature_discriminator.state_dict(),
            "decoder": decoder.state_dict(),
            "image_discriminator": image_discriminator.state_dict(),
            "ema_victim_client": ema.client.state_dict() if ema is not None else None,
            "ema_victim_server": ema.server.state_dict() if ema is not None else None,
            "summary": summary,
        },
        out_dir / "full_sdar_last.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
