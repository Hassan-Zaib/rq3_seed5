from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tinyimagenet_class_to_idx(root):
    root = Path(root)
    wnids_file = root / "wnids.txt"
    if wnids_file.exists():
        wnids = [line.strip() for line in wnids_file.read_text().splitlines() if line.strip()]
    else:
        wnids = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    return {wnid: idx for idx, wnid in enumerate(wnids)}


class TinyImageNetDataset(Dataset):
    def __init__(self, root, split="train", train=True):
        self.root = Path(root)
        self.split = split
        self.class_to_idx = tinyimagenet_class_to_idx(root)
        self.raw_transform = transforms.ToTensor()
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(num_ops=2, magnitude=9),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        self.samples = self._load_samples()

    def _load_samples(self):
        if self.split == "train":
            samples = []
            train_root = self.root / "train"
            for wnid, class_idx in self.class_to_idx.items():
                img_dir = train_root / wnid / "images"
                for path in sorted(img_dir.glob("*.JPEG")):
                    samples.append((path, class_idx))
            return samples

        if self.split == "val":
            ann = self.root / "val" / "val_annotations.txt"
            img_root = self.root / "val" / "images"
            samples = []
            for line in ann.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                samples.append((img_root / filename, self.class_to_idx[wnid]))
            return samples

        raise ValueError(f"Unsupported split {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        raw = self.raw_transform(image)
        return x, int(label), raw, str(path), int(index)


def build_loaders(root, batch_size=128, num_workers=8):
    train_ds = TinyImageNetDataset(root, split="train", train=True)
    val_ds = TinyImageNetDataset(root, split="val", train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_eval_loader(root, split="val", batch_size=128, num_workers=8):
    ds = TinyImageNetDataset(root, split=split, train=False)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def inverse_normalize(x):
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)
