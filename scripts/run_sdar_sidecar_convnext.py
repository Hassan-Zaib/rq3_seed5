#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convnext_split import build_split_convnext_tiny, expected_activation_shape
from data_tinyimagenet import TinyImageNetDataset, build_eval_loader, build_loaders
from full_sdar import (
    ConditionalDecoder,
    ConditionalFeatureDiscriminator,
    ConditionalImageDiscriminator,
    set_trainable,
)
from perturbations import apply_perturbation
from run_sdar_on_convnext_smashed import (
    backward_and_step,
    maybe_lpips,
    next_batch,
    reconstruct_fixed_probe,
)
from train_split_convnext import (
    ModelEmaPair,
    SoftTargetCrossEntropy,
    apply_mixup_cutmix,
    evaluate,
)
from train_utils import (
    AverageMeter,
    accuracy_top1,
    get_device,
    reconstruction_metric_summary,
    save_image_grid,
    save_json,
    set_seed,
)


def build_clean_attack_loaders(data_root, batch_size, num_workers, seed):
    clean_train = TinyImageNetDataset(data_root, split="train", train=False)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(clean_train), generator=generator).tolist()
    middle = len(indices) // 2
    observer = Subset(clean_train, indices[:middle])
    auxiliary = Subset(clean_train, indices[middle:])
    common = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return DataLoader(observer, **common), DataLoader(auxiliary, **common)


def eval_decoder(decoder, client, loader, device, defense, max_samples):
    decoder.eval()
    client.eval()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--run-root", default="results_c1_sidecar_cutmix_sdar")
    parser.add_argument("--fixed-probe-dir", default="fixed_probe")
    parser.add_argument("--cut", default="c1", choices=["c1", "c2", "c3", "c4"])
    parser.add_argument("--defense", default="none", choices=["none", "q8", "q4", "q2"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--attack-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-switch-prob", type=float, default=0.5)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--model-ema-decay", type=float, default=0.9999)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--e-lr", type=float, default=1e-3)
    parser.add_argument("--d-lr", type=float, default=5e-4)
    parser.add_argument("--e-dis-lr", type=float, default=2e-5)
    parser.add_argument("--d-dis-lr", type=float, default=5e-9)
    parser.add_argument("--lambda1", type=float, default=0.02)
    parser.add_argument("--lambda2", type=float, default=1e-5)
    parser.add_argument("--attack-updates-per-batch", type=int, default=1)
    parser.add_argument("--require-lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.run_root) / args.defense / args.cut / "sidecar_sdar"
    victim_dir = Path(args.run_root) / args.defense / args.cut / "victim"
    out_dir.mkdir(parents=True, exist_ok=True)
    victim_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args.data_root, args.batch_size, args.num_workers)
    observer_loader, auxiliary_loader = build_clean_attack_loaders(
        args.data_root, args.attack_batch_size, args.num_workers, args.split_seed
    )
    observer_iterator = iter(observer_loader)
    auxiliary_iterator = iter(auxiliary_loader)

    client, server = build_split_convnext_tiny(
        args.cut, num_classes=200, drop_path_rate=args.drop_path_rate
    )
    client.to(device)
    server.to(device)
    with torch.no_grad():
        z_shape = tuple(client(torch.zeros(2, 3, 64, 64, device=device)).shape)
    expected = (2,) + expected_activation_shape(args.cut)
    if z_shape != expected:
        raise RuntimeError(f"Shape mismatch: got {z_shape}, expected {expected}")
    print(f"{args.cut.upper()} activation shape: {z_shape}, expected {expected}", flush=True)

    channels, height, _ = expected_activation_shape(args.cut)
    simulator, _ = build_split_convnext_tiny(
        args.cut, num_classes=200, drop_path_rate=args.drop_path_rate
    )
    simulator = simulator.to(device)
    feature_discriminator = ConditionalFeatureDiscriminator(channels, height, 200).to(device)
    decoder = ConditionalDecoder(channels, height, 200, out_hw=64).to(device)
    image_discriminator = ConditionalImageDiscriminator(200, image_hw=64).to(device)

    victim_criterion = SoftTargetCrossEntropy()
    ce_hard = nn.CrossEntropyLoss()
    adversarial_loss = nn.BCEWithLogitsLoss()
    victim_optimizer = AdamW(
        list(client.parameters()) + list(server.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.warmup_epochs > 0 and args.epochs > args.warmup_epochs:
        warmup = LinearLR(victim_optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(victim_optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(victim_optimizer, [warmup, cosine], [args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(victim_optimizer, T_max=args.epochs)
    ema = ModelEmaPair(client, server, decay=args.model_ema_decay)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=amp_enabled and args.amp_dtype == "fp16")

    e_optimizer = Adam(simulator.parameters(), lr=args.e_lr)
    e_dis_optimizer = Adam(feature_discriminator.parameters(), lr=args.e_dis_lr)
    d_optimizer = Adam(decoder.parameters(), lr=args.d_lr)
    d_dis_optimizer = Adam(image_discriminator.parameters(), lr=args.d_dis_lr)

    history = []
    attack_history = {key: [] for key in (
        "eg_loss", "eg_acc", "e_loss", "e_dis_loss", "d_loss", "d_dis_loss", "attack_loss"
    )}
    best_acc = -1.0
    started = time.time()

    config = vars(args).copy()
    config["activation_shape"] = z_shape
    config["expected_activation_shape"] = expected
    config["protocol"] = "sidecar_sdar_victim_trains_with_convnext_recipe_attacker_observes_clean_batches"
    save_json(out_dir / "config.json", config)

    for epoch in range(1, args.epochs + 1):
        client.train()
        server.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for x, y, _, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mixed_x, soft_y = apply_mixup_cutmix(x, y, args)
            victim_optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                z = apply_perturbation(client(mixed_x), args.defense)
                logits = server(z)
                loss = victim_criterion(logits, soft_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite victim loss at epoch {epoch}")
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(victim_optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(client.parameters()) + list(server.parameters()), args.grad_clip
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite victim grad norm at epoch {epoch}")
            scaler.step(victim_optimizer)
            scaler.update()
            ema.update(client, server)
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(accuracy_top1(logits.detach(), soft_y.argmax(dim=1)), x.size(0))

            for _ in range(args.attack_updates_per_batch):
                observer_batch, observer_iterator = next_batch(observer_iterator, observer_loader)
                auxiliary_batch, auxiliary_iterator = next_batch(auxiliary_iterator, auxiliary_loader)
                xo, yo, raw_o = observer_batch[0].to(device), observer_batch[1].to(device), observer_batch[2].to(device)
                xa, ya, raw_a = auxiliary_batch[0].to(device), auxiliary_batch[1].to(device), auxiliary_batch[2].to(device)

                client.eval()
                server.eval()
                with torch.no_grad():
                    z_real = apply_perturbation(client(xo), args.defense).detach()

                set_trainable(server, False)
                set_trainable(feature_discriminator, False)
                e_optimizer.zero_grad(set_to_none=True)
                z_sim = apply_perturbation(simulator(xa), args.defense)
                sim_logits = server(z_sim)
                eg_loss = ce_hard(sim_logits, ya)
                e_gen_loss = adversarial_loss(
                    feature_discriminator(z_sim, ya),
                    torch.ones(args.attack_batch_size, device=device),
                )
                e_loss = eg_loss + args.lambda1 * e_gen_loss
                backward_and_step(e_loss, e_optimizer, simulator.parameters(), "simulator")
                set_trainable(feature_discriminator, True)
                set_trainable(server, True)

                e_dis_optimizer.zero_grad(set_to_none=True)
                real_logits = feature_discriminator(z_real, yo)
                fake_logits = feature_discriminator(z_sim.detach(), ya)
                e_dis_loss = adversarial_loss(
                    real_logits, torch.ones(args.attack_batch_size, device=device)
                ) + adversarial_loss(
                    fake_logits, torch.zeros(args.attack_batch_size, device=device)
                )
                backward_and_step(
                    e_dis_loss, e_dis_optimizer, feature_discriminator.parameters(), "feature_discriminator"
                )

                set_trainable(image_discriminator, False)
                d_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z_sim_updated = apply_perturbation(simulator(xa), args.defense)
                decoded_aux = decoder(z_sim_updated.detach(), ya)
                d_mse = torch.mean((raw_a - decoded_aux) ** 2)
                decoded_obs = decoder(z_real, yo)
                d_gen = adversarial_loss(
                    image_discriminator(decoded_obs, yo),
                    torch.ones(args.attack_batch_size, device=device),
                )
                d_loss = d_mse + args.lambda2 * d_gen
                backward_and_step(d_loss, d_optimizer, decoder.parameters(), "decoder")
                set_trainable(image_discriminator, True)

                d_dis_optimizer.zero_grad(set_to_none=True)
                d_dis_loss = adversarial_loss(
                    image_discriminator(raw_a, ya),
                    torch.ones(args.attack_batch_size, device=device),
                ) + adversarial_loss(
                    image_discriminator(decoded_obs.detach(), yo),
                    torch.zeros(args.attack_batch_size, device=device),
                )
                backward_and_step(
                    d_dis_loss, d_dis_optimizer, image_discriminator.parameters(), "image_discriminator"
                )

                with torch.no_grad():
                    attack_loss = torch.mean((raw_o - decoder(z_real, yo)) ** 2)
                    attack_history["eg_loss"].append(float(eg_loss.item()))
                    attack_history["eg_acc"].append(float((sim_logits.argmax(dim=1) == ya).float().mean().item()))
                    attack_history["e_loss"].append(float(e_loss.item()))
                    attack_history["e_dis_loss"].append(float(e_dis_loss.item()))
                    attack_history["d_loss"].append(float(d_loss.item()))
                    attack_history["d_dis_loss"].append(float(d_dis_loss.item()))
                    attack_history["attack_loss"].append(float(attack_loss.item()))

        scheduler.step()
        raw_val = evaluate(client, server, val_loader, device, args.defense)
        ema_val = evaluate(ema.client, ema.server, val_loader, device, args.defense)
        if ema_val["val_acc"] >= raw_val["val_acc"]:
            eval_client, eval_server, val, val_source = ema.client, ema.server, ema_val, "ema"
        else:
            eval_client, eval_server, val, val_source = client, server, raw_val, "raw"
        row = {
            "epoch": epoch,
            "train_loss": loss_meter.avg,
            "train_acc": acc_meter.avg,
            "raw_val_acc": raw_val["val_acc"],
            "ema_val_acc": ema_val["val_acc"],
            "val_acc": val["val_acc"],
            "val_source": val_source,
            "lr": scheduler.get_last_lr()[0],
            "attack_loss_last100": float(np.mean(attack_history["attack_loss"][-100:])) if attack_history["attack_loss"] else None,
        }
        history.append(row)
        print(
            f"[sidecar {args.defense} {args.cut}] epoch {epoch:03d}/{args.epochs} "
            f"train_acc={row['train_acc']:.4f} raw_val={row['raw_val_acc']:.4f} "
            f"ema_val={row['ema_val_acc']:.4f} best_eval={row['val_acc']:.4f}({val_source}) "
            f"attack={row['attack_loss_last100']:.4f}",
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
        torch.save(ckpt, victim_dir / "checkpoint_last.pt")
        if row["val_acc"] > best_acc:
            best_acc = row["val_acc"]
            torch.save(ckpt, victim_dir / "checkpoint_best.pt")
        save_json(victim_dir / "history.json", history)

    best_checkpoint = torch.load(victim_dir / "checkpoint_best.pt", map_location=device)
    client.load_state_dict(best_checkpoint["client"])
    server.load_state_dict(best_checkpoint["server"])
    client.eval()
    server.eval()
    validation_loader = build_eval_loader(
        args.data_root, split="val", batch_size=args.attack_batch_size, num_workers=args.num_workers
    )
    val = evaluate(client, server, validation_loader, device, args.defense)
    originals, reconstructions, labels = eval_decoder(
        decoder, client, validation_loader, device, args.defense, args.eval_samples
    )
    np.save(out_dir / "originals.npy", originals.astype(np.float32))
    np.save(out_dir / "reconstructions.npy", reconstructions.astype(np.float32))
    np.save(out_dir / "labels.npy", labels.astype(np.int64))
    np.save(out_dir / "sidecar_sdar_history.npy", attack_history, allow_pickle=True)
    save_image_grid(
        out_dir / "original_recon_grid.png",
        [originals[:20], reconstructions[:20]],
        ["Original", "Sidecar SDAR"],
    )
    fixed = reconstruct_fixed_probe(decoder, client, device, args.defense, args.fixed_probe_dir, out_dir)
    lpips_model = maybe_lpips(device, args.require_lpips, args.compute_lpips)
    metrics = reconstruction_metric_summary(originals, reconstructions, lpips_model, device)
    if fixed is not None:
        fixed_metrics = reconstruction_metric_summary(fixed[0], fixed[1], lpips_model, device)
        metrics.update({f"fixed_probe_{key}": value for key, value in fixed_metrics.items()})

    summary = {
        "protocol": "sidecar_sdar_while_training",
        "cut": args.cut,
        "defense": args.defense,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "attack_batch_size": args.attack_batch_size,
        "best_val_acc": best_acc,
        "final_val_acc": val["val_acc"],
        "final_raw_val_acc": history[-1]["raw_val_acc"],
        "final_ema_val_acc": history[-1]["ema_val_acc"],
        "final_train_acc": history[-1]["train_acc"],
        "attack_updates": int(len(attack_history["attack_loss"])),
        "attack_loss_last100": float(np.mean(attack_history["attack_loss"][-100:])),
        "eg_acc_last100": float(np.mean(attack_history["eg_acc"][-100:])),
        "total_runtime_sec": time.time() - started,
        **metrics,
    }
    save_json(out_dir / "sidecar_sdar_summary.json", summary)
    save_json(victim_dir / "victim_summary.json", {
        "cut": args.cut,
        "defense": args.defense,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_acc": best_acc,
        "final_val_acc": val["val_acc"],
        "final_train_acc": history[-1]["train_acc"],
        "history": history,
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
