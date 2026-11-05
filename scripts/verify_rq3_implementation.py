#!/usr/bin/env python
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convnext_split import build_split_convnext_tiny, expected_activation_shape
from full_sdar import (
    ConditionalDecoder,
    ConditionalFeatureDiscriminator,
    ConditionalImageDiscriminator,
)
from perturbations import apply_perturbation


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verification device: {device}")

    for cut in ("c1", "c2", "c3", "c4"):
        client, server = build_split_convnext_tiny(
            cut, num_classes=200, drop_path_rate=0.1
        )
        client.to(device).train()
        server.to(device).train()
        client_ids = {id(parameter) for parameter in client.parameters()}
        server_ids = {id(parameter) for parameter in server.parameters()}
        require(client_ids.isdisjoint(server_ids), f"{cut}: client/server share parameters")

        x = torch.randn(2, 3, 64, 64, device=device)
        labels = torch.tensor([0, 199], device=device)
        z = client(x)
        require(
            tuple(z.shape) == (2,) + expected_activation_shape(cut),
            f"{cut}: unexpected activation shape {tuple(z.shape)}",
        )
        logits = server(z)
        require(tuple(logits.shape) == (2, 200), f"{cut}: invalid logits")

        for defense, levels in (("q8", 256), ("q4", 16), ("q2", 4)):
            defended = apply_perturbation(z, defense)
            require(torch.isfinite(defended).all(), f"{cut}/{defense}: non-finite z")
            dims = tuple(range(1, defended.ndim))
            z_min = z.amin(dim=dims, keepdim=True)
            z_max = z.amax(dim=dims, keepdim=True)
            scale = (z_max - z_min).clamp_min(1e-8)
            codes = torch.round(
                (defended - z_min) / scale * (levels - 1)
            ).to(torch.int64)
            require(codes.min() >= 0, f"{cut}/{defense}: code below zero")
            require(codes.max() < levels, f"{cut}/{defense}: code above range")
            reconstructed = codes.to(defended.dtype) / (levels - 1) * scale + z_min
            require(
                torch.allclose(defended, reconstructed, atol=1e-6, rtol=1e-6),
                f"{cut}/{defense}: forward output is not exactly dequantized",
            )

        z.retain_grad()
        loss = nn.CrossEntropyLoss()(server(apply_perturbation(z, "q2")), labels)
        loss.backward()
        require(z.grad is not None, f"{cut}: STE did not return a gradient")
        require(torch.isfinite(z.grad).all(), f"{cut}: non-finite STE gradient")

        channels, height, _ = expected_activation_shape(cut)
        decoder = ConditionalDecoder(channels, height, 200).to(device)
        feature_discriminator = ConditionalFeatureDiscriminator(
            channels, height, 200
        ).to(device)
        image_discriminator = ConditionalImageDiscriminator(200).to(device)
        reconstruction = decoder(z.detach(), labels)
        require(
            tuple(reconstruction.shape) == (2, 3, 64, 64),
            f"{cut}: decoder shape mismatch",
        )
        require(
            tuple(feature_discriminator(z.detach(), labels).shape) == (2,),
            f"{cut}: feature discriminator shape mismatch",
        )
        require(
            tuple(image_discriminator(reconstruction, labels).shape) == (2,),
            f"{cut}: image discriminator shape mismatch",
        )
        print(f"{cut.upper()}: architecture, split, STE, and SDAR modules passed")

        del client, server, decoder, feature_discriminator, image_discriminator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("RQ3 implementation verification passed.")


if __name__ == "__main__":
    main()
