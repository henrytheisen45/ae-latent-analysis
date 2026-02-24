# ae_latent/data/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any, Sequence, Literal, Dict

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

SplitName = Literal["train", "val", "test"]


# ----------------------------
# Index / determinism helpers
# ----------------------------

def seed_worker(worker_id: int) -> None:
    """
    Ensure each dataloader worker has a deterministic RNG state.
    PyTorch sets a base seed per worker; we derive numpy/random from it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def base_index(ds: Dataset, idx: int) -> int:
    """
    Map an index into an arbitrarily nested Dataset/Subset/random_split view
    back to the base dataset index (torchvision dataset index).

    Works for chains like:
      Subset(Subset(CIFAR10, ...), ...) and random_split(...) outputs.
    """
    while isinstance(ds, Subset):
        idx = int(ds.indices[idx])
        ds = ds.dataset
    return int(idx)


class IndexedDataset(Dataset):
    """
    Wrap a dataset so __getitem__ returns (x, y, sid) where sid is a globally
    unique identifier for the sample across dataset + split.

    sid format: (dataset_name, split_name, base_idx)

    Notes:
    - For datasets where train/test are distinct underlying objects (e.g. CelebA),
      base_idx is only meaningful within the split, hence the split_name namespace.
    - This makes sid collision-proof when you later merge/compare embeddings across splits.
    """
    def __init__(self, view: Dataset, *, dataset_name: str, split_name: SplitName):
        self.view = view
        self.dataset_name = str(dataset_name)
        self.split_name: SplitName = split_name

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, idx: int):
        x, y = self.view[idx]
        bidx = base_index(self.view, idx)
        sid = (self.dataset_name, self.split_name, bidx)
        return x, y, sid


def make_collate_fn(*, return_ids: bool) -> Optional[Callable[[Sequence[Any]], Any]]:
    """
    Collate function that optionally drops the sid from (x, y, sid).

    return_ids=False -> batches become (X, Y)
    return_ids=True  -> default collation (X, Y, SID)
    """
    if return_ids:
        return None

    def _collate_drop_ids(batch: Sequence[Any]):
        xs, ys, _sids = zip(*batch)
        X = torch.utils.data.default_collate(xs)
        Y = torch.utils.data.default_collate(ys)
        return X, Y

    return _collate_drop_ids


# ----------------------------
# Dataset info + transforms
# ----------------------------

@dataclass(frozen=True)
class DatasetInfo:
    name: str
    channels: int
    img_size: int
    num_classes: Optional[int]
    target_kind: str  # "class" | "attrs" | "none"
    target_dim: Optional[int]


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    name = dataset_name.lower()
    configs = {
        "mnist":         dict(channels=1, img_size=32, num_classes=10,  target_kind="class", target_dim=None),
        "fashion_mnist": dict(channels=1, img_size=32, num_classes=10,  target_kind="class", target_dim=None),
        "cifar10":       dict(channels=3, img_size=32, num_classes=10,  target_kind="class", target_dim=None),
        "cifar100":      dict(channels=3, img_size=32, num_classes=100, target_kind="class", target_dim=None),
        "svhn":          dict(channels=3, img_size=32, num_classes=10,  target_kind="class", target_dim=None),
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


def make_transform(info: DatasetInfo) -> transforms.Compose:
    """
    Deterministic preprocessing:
      - Resize -> CenterCrop -> ToTensor -> Normalize to [-1,1]
    This matches out_activation='tanh' expectation.
    """
    if info.channels == 1:
        return transforms.Compose([
            transforms.Resize(info.img_size),
            transforms.CenterCrop(info.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    return transforms.Compose([
        transforms.Resize(info.img_size),
        transforms.CenterCrop(info.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_base_datasets(
    dataset_name: str,
    data_root: str,
    download: bool = True,
) -> Tuple[Dataset, Dataset, DatasetInfo]:
    info = get_dataset_info(dataset_name)
    tfm = make_transform(info)

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
        test  = datasets.SVHN(root=data_root, split="test",  transform=tfm, download=download)

    elif name == "celeba":
        # Enforce attrs explicitly so DatasetInfo matches.
        train = datasets.CelebA(root=data_root, split="train", transform=tfm, target_type="attr", download=download)
        test  = datasets.CelebA(root=data_root, split="test",  transform=tfm, target_type="attr", download=download)

    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return train, test, info


# ----------------------------
# Core builder (parameterized)
# ----------------------------

def get_dataloaders(
    dataset_name: str,
    *,
    batch_size: int,
    num_workers: int,
    data_root: str,
    val_split: float,
    seed: int,
    subset_size: Optional[int],
    subset_seed: Optional[int],
    return_ids: bool,
    shuffle_train: bool,
    shuffle_val: bool,
    shuffle_test: bool,
    pin_memory: bool,
    persistent_workers: bool,
    drop_last: bool,
    download: bool,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, DatasetInfo]:
    """
    Returns (train_loader, val_loader_or_None, test_loader, dataset_info)

    Guarantees:
    - Deterministic subset selection (optional) using subset_seed (defaults to seed)
    - Deterministic train/val split using seed
    - Deterministic worker seeding (numpy/random)
    - Deterministic shuffling when shuffle_* is True (via DataLoader(generator=...))
    - Optional stable, collision-proof sample IDs via return_ids

    Batch format:
    - return_ids=False (default): (x, y)
    - return_ids=True:            (x, y, sid)
      where sid = (dataset_name, split_name, base_idx)
    """
    base_train, base_test, info = load_base_datasets(dataset_name, data_root, download=download)

    # Optional subset on the *train* pool (before train/val split)
    if subset_size is not None:
        subset_size = int(subset_size)
        if subset_size <= 0 or subset_size > len(base_train):
            raise ValueError(f"subset_size={subset_size} invalid for train length {len(base_train)}")
        ss = seed if subset_seed is None else int(subset_seed)
        rng = np.random.default_rng(ss)
        indices = rng.choice(len(base_train), size=subset_size, replace=False).tolist()
        base_train = Subset(base_train, indices)

    # Validate split (allow 0.0 -> no val)
    if not (0.0 <= val_split < 1.0):
        raise ValueError(f"val_split must be in [0,1). Got {val_split}")

    # Split train/val if requested
    if val_split > 0.0:
        n_total = len(base_train)
        n_val = int(round(val_split * n_total))
        n_val = max(1, min(n_val, n_total - 1))
        n_train = n_total - n_val

        train_view, val_view = random_split(
            base_train,
            lengths=[n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_view, val_view = base_train, None

    # Wrap for stable IDs (we always wrap, and collate drops IDs when return_ids=False)
    train_ds = IndexedDataset(train_view, dataset_name=info.name, split_name="train")
    val_ds = IndexedDataset(val_view, dataset_name=info.name, split_name="val") if val_view is not None else None
    test_ds = IndexedDataset(base_test, dataset_name=info.name, split_name="test")

    # Generators:
    # - always pass generator for deterministic shuffling
    g_train = torch.Generator().manual_seed(seed)
    g_eval  = torch.Generator().manual_seed(seed + 1)

    use_persistent = bool(persistent_workers and num_workers > 0)
    collate_fn = make_collate_fn(return_ids=return_ids)
    worker_init = seed_worker if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init,
        generator=g_train if shuffle_train else None,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )

    val_loader: Optional[DataLoader] = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init,
            generator=g_eval if shuffle_val else None,
            persistent_workers=use_persistent,
            collate_fn=collate_fn,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init,
        generator=g_eval if shuffle_test else None,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, info


# ----------------------------
# Config-driven entrypoint (what you actually want)
# ----------------------------

def build_dataloaders(cfg: dict) -> dict:
    """
    Build dataloaders from experiment JSON root config.

    Expects:
      cfg["dataset"], cfg["seed"], cfg["data"] (dict)

    Returns:
      {"train": ..., "val": ..., "test": ..., "info": DatasetInfo}
    """
    if "dataset" not in cfg:
        raise ValueError("cfg missing required key 'dataset'")
    if "seed" not in cfg:
        raise ValueError("cfg missing required key 'seed'")
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        raise ValueError("cfg missing required dict key 'data'")

    d = cfg["data"]
    train_loader, val_loader, test_loader, info = get_dataloaders(
        cfg["dataset"],
        batch_size=int(d["batch_size"]),
        num_workers=int(d.get("num_workers", 0)),
        data_root=str(d["data_root"]),
        val_split=float(d.get("val_split", 0.1)),
        seed=int(cfg["seed"]),
        subset_size=d.get("subset_size", None),
        subset_seed=d.get("subset_seed", None),
        return_ids=bool(d.get("return_ids", False)),
        shuffle_train=bool(d.get("shuffle_train", True)),
        shuffle_val=bool(d.get("shuffle_val", False)),
        shuffle_test=bool(d.get("shuffle_test", False)),
        pin_memory=bool(d.get("pin_memory", True)),
        persistent_workers=bool(d.get("persistent_workers", True)),
        drop_last=bool(d.get("drop_last", False)),
        download=bool(d.get("download", True)),
    )

    out = {"train": train_loader, "test": test_loader, "info": info}
    if val_loader is not None:
        out["val"] = val_loader
    return out