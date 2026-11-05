import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def accuracy_top1(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.total / max(1, self.count)


def mse_to_psnr(mse):
    mse = float(mse)
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def tensor_to_uint8_image(x):
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim == 3:
        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")


def array_to_uint8_image(x):
    if isinstance(x, torch.Tensor):
        return tensor_to_uint8_image(x)
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_image_grid(path, rows, row_labels=None, cell_size=84, pad=8, label_width=120):
    """Save a simple labeled grid from rows of CHW tensors or numpy arrays in [0,1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0 or len(rows[0]) == 0:
        raise ValueError("save_image_grid needs at least one image")

    row_labels = row_labels or ["" for _ in rows]
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    width = label_width + n_cols * cell_size + (n_cols + 1) * pad
    height = n_rows * cell_size + (n_rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for r, row in enumerate(rows):
        y = pad + r * (cell_size + pad)
        if row_labels[r]:
            draw.text((10, y + cell_size // 3), row_labels[r], fill="black", font=font)
        for c, image in enumerate(row):
            x = label_width + pad + c * (cell_size + pad)
            img = array_to_uint8_image(image).resize((cell_size, cell_size), Image.BICUBIC)
            canvas.paste(img, (x, y))
    canvas.save(path)


def standard_ssim_batch(x, y):
    """Standard windowed SSIM, matching the publication metric rather than a global proxy."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "Standard SSIM requires scikit-image. Install it with: pip install scikit-image"
        ) from exc

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    values = []
    for original, reconstruction in zip(x, y):
        original = np.transpose(original, (1, 2, 0))
        reconstruction = np.transpose(reconstruction, (1, 2, 0))
        values.append(
            structural_similarity(
                original,
                reconstruction,
                data_range=1.0,
                channel_axis=2,
                win_size=11,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        )
    return np.asarray(values, dtype=np.float32)


def reconstruction_metric_summary(originals, reconstructions, lpips_model=None, device=None, batch_size=64):
    originals = np.clip(np.asarray(originals, dtype=np.float32), 0, 1)
    reconstructions = np.clip(np.asarray(reconstructions, dtype=np.float32), 0, 1)
    per_mse = ((originals - reconstructions) ** 2).mean(axis=(1, 2, 3))
    per_psnr = np.array([mse_to_psnr(v) for v in per_mse], dtype=np.float32)
    per_ssim = standard_ssim_batch(originals, reconstructions)
    out = {
        "n": int(len(originals)),
        "mse_mean": float(per_mse.mean()),
        "mse_std": float(per_mse.std()),
        "mse_min": float(per_mse.min()),
        "mse_max": float(per_mse.max()),
        "psnr_mean": float(per_psnr.mean()),
        "psnr_std": float(per_psnr.std()),
        "psnr_min": float(per_psnr.min()),
        "psnr_max": float(per_psnr.max()),
        "ssim_mean": float(per_ssim.mean()),
        "ssim_std": float(per_ssim.std()),
        "ssim_min": float(per_ssim.min()),
        "ssim_max": float(per_ssim.max()),
        "asr_ssim_0.90": float((per_ssim >= 0.90).mean()),
    }

    if lpips_model is not None:
        vals = []
        device = device or get_device()
        with torch.no_grad():
            for start in range(0, len(originals), batch_size):
                o = torch.from_numpy(originals[start : start + batch_size]).to(device)
                r = torch.from_numpy(reconstructions[start : start + batch_size]).to(device)
                o = o * 2 - 1
                r = r * 2 - 1
                vals.append(lpips_model(o, r).view(-1).detach().cpu().numpy())
        per_lpips = np.concatenate(vals, axis=0)
        out["lpips_mean"] = float(per_lpips.mean())
        out["lpips_std"] = float(per_lpips.std())
        out["lpips_min"] = float(per_lpips.min())
        out["lpips_max"] = float(per_lpips.max())
    else:
        out["lpips_mean"] = None
        out["lpips_std"] = None
        out["lpips_min"] = None
        out["lpips_max"] = None

    return out
