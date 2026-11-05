import torch


class _QuantizeDequantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, bits):
        levels = (1 << int(bits)) - 1
        dims = tuple(range(1, z.ndim))
        z_min = z.amin(dim=dims, keepdim=True)
        z_max = z.amax(dim=dims, keepdim=True)
        scale = (z_max - z_min).clamp_min(1e-8)
        codes = torch.round((z - z_min) / scale * levels)
        return codes / levels * scale + z_min

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize_dequantize_ste(z, bits):
    if bits is None:
        return z
    return _QuantizeDequantizeSTE.apply(z, int(bits))


def gaussian_noise(z, sigma):
    if sigma is None or sigma <= 0:
        return z
    return z + torch.randn_like(z) * float(sigma)


def apply_perturbation(z, defense):
    defense = defense.lower()
    if defense == "none":
        return z
    if defense.startswith("q"):
        return quantize_dequantize_ste(z, int(defense[1:]))
    if defense.startswith("noise"):
        sigma = float(defense.replace("noise", ""))
        return gaussian_noise(z, sigma)
    raise ValueError(f"Unsupported defense {defense}")


def defense_label(defense):
    defense = defense.lower()
    if defense == "none":
        return "none"
    if defense.startswith("q"):
        return defense
    if defense.startswith("noise"):
        return defense
    raise ValueError(f"Unsupported defense {defense}")
