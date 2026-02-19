from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


# ----------------------------
# Small helpers
# ----------------------------

class IndexedDataset(Dataset):
    """
    Wrap a dataset so __getitem__ returns (x, y, idx) where idx is the index
    into the *wrapped* dataset (after any Subset / split).
    """
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, idx


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    channels: int
    img_size: int
    num_classes: Optional[int]
    target_kind: str  # "class" | "attrs" | "none"
    target_dim: Optional[int]  # e.g. 40 for CelebA attrs


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    name = dataset_name.lower()
    configs = {
        "mnist":         dict(channels=1, img_size=28, num_classes=10,  target_kind="class", target_dim=1),
        "fashion_mnist": dict(channels=1, img_size=28, num_classes=10,  target_kind="class", target_dim=1),
        "cifar10":       dict(channels=3, img_size=32, num_classes=10,  target_kind="class", target_dim=1),
        "cifar100":      dict(channels=3, img_size=32, num_classes=100, target_kind="class", target_dim=1),
        "svhn":          dict(channels=3, img_size=32, num_classes=10,  target_kind="class", target_dim=1),
        "celeba":        dict(channels=3, img_size=64, num_classes=None, target_kind="attrs", target_dim=40),
    }
    if name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    c = configs[name]
    return DatasetInfo(
        name=name,
        channels=c["channels"],
        img_size=c["img_size"],
        num_classes=c["num_classes"],
        target_kind=c["target_kind"],
        target_dim=c["target_dim"],
    )


def _make_transform(info: DatasetInfo) -> transforms.Compose:
    # All your datasets use Normalize(mean=0.5, std=0.5) => maps [0,1] to [-1,1]
    if info.channels == 1:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    # RGB
    if info.name == "celeba":
        return transforms.Compose([
            transforms.Resize(info.img_size),
            transforms.CenterCrop(info.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _load_base_datasets(
    dataset_name: str,
    data_root: str,
    download: bool = True,
) -> Tuple[Dataset, Dataset, DatasetInfo]:
    info = get_dataset_info(dataset_name)
    tfm = _make_transform(info)

    name = info.name
    if name == "mnist":
        train = datasets.MNIST(root=data_root, train=True, transform=tfm, download=download)
        test  = datasets.MNIST(root=data_root, train=False, transform=tfm, download=download)
    elif name == "fashion_mnist":
        train = datasets.FashionMNIST(root=data_root, train=True, transform=tfm, download=download)
        test  = datasets.FashionMNIST(root=data_root, train=False, transform=tfm, download=download)
    elif name == "cifar10":
        train = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=download)
        test  = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=download)
    elif name == "cifar100":
        train = datasets.CIFAR100(root=data_root, train=True, transform=tfm, download=download)
        test  = datasets.CIFAR100(root=data_root, train=False, transform=tfm, download=download)
    elif name == "svhn":
        train = datasets.SVHN(root=data_root, split="train", transform=tfm, download=download)
        test  = datasets.SVHN(root=data_root, split="test", transform=tfm, download=download)
    elif name == "celeba":
        train = datasets.CelebA(root=data_root, split="train", transform=tfm, download=download)
        test  = datasets.CelebA(root=data_root, split="test",  transform=tfm, download=download)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return train, test, info


# ----------------------------
# Main entrypoint
# ----------------------------

def get_dataloaders(
    dataset_name: str,
    *,
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data",
    val_split: float = 0.1,
    seed: int = 42,
    subset_size: Optional[int] = None,
    subset_seed: Optional[int] = None,
    analysis_mode: bool = False,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """
    Returns train_loader, val_loader, test_loader, dataset_info

    Key behaviors:
    - Deterministic splitting via `seed`
    - Optional deterministic subset selection via `subset_size` and `subset_seed`
    - analysis_mode=True:
        - shuffle=False everywhere
        - datasets are wrapped to return (x, y, idx) so latents/attrs stay aligned
    """
    base_train, base_test, info = _load_base_datasets(dataset_name, data_root, download=download)

    # Optional subset on the *train* pool (before train/val split)
    if subset_size is not None and subset_size < len(base_train):
        ss = seed if subset_seed is None else subset_seed
        rng = np.random.default_rng(ss)
        indices = rng.choice(len(base_train), size=subset_size, replace=False)
        base_train = Subset(base_train, indices)

    # Train/val split
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1). Got {val_split}")

    n_total = len(base_train)
    n_val = int(round(val_split * n_total))
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError(f"val_split={val_split} yields empty train/val (n_total={n_total}).")

    train_ds, val_ds = random_split(
        base_train,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # In analysis mode, preserve indices and deterministic ordering
    if analysis_mode:
        train_ds = IndexedDataset(train_ds)
        val_ds   = IndexedDataset(val_ds)
        test_ds  = IndexedDataset(base_test)
        shuffle_train = False
        shuffle_val = False
        shuffle_test = False
    else:
        test_ds = base_test
        shuffle_train = True
        shuffle_val = False
        shuffle_test = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=shuffle_val,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=shuffle_test,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, info
