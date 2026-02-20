from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any, Sequence, Literal

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

SplitName = Literal["train", "val", "test"]


# ----------------------------
# Index / determinism helpers
# ----------------------------

def _seed_worker(worker_id: int) -> None:
    """
    Ensure each dataloader worker has a deterministic RNG state.
    PyTorch sets a base seed per worker; we derive numpy/random from it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _base_index(ds: Dataset, idx: int) -> int:
    """
    Map an index into an arbitrarily nested Dataset/Subset/random_split view
    back to the basew dataset index (torchvision dataset index).

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
        base_idx = _base_index(self.view, idx)
        sid = (self.dataset_name, self.split_name, base_idx)
        return x, y, sid


def _make_collate_fn(*, return_ids: bool) -> Optional[Callable[[Sequence[Any]], Any]]:
    """
    Collate function that optionally drops the sid from (x, y, sid).

    return_ids=False, batches become (X, Y).
    return_ids=True, use default collation (X, Y, SID).
    """
    if return_ids:
        return None  

    def _collate_drop_ids(batch: Sequence[Any]):
        # Batch elements are expected to be (x, y, sid)
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
        "mnist":         dict(channels=1, img_size=28, num_classes=10,  target_kind="class", target_dim=None),
        "fashion_mnist": dict(channels=1, img_size=28, num_classes=10,  target_kind="class", target_dim=None),
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


def _make_transform(info: DatasetInfo) -> transforms.Compose:
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
        test  = datasets.SVHN(root=data_root, split="test",  transform=tfm, download=download)

    elif name == "celeba":
        # Enforce attrs explicitly so DatasetInfo matches
        train = datasets.CelebA(root=data_root, split="train", transform=tfm, target_type="attr", download=download)
        test  = datasets.CelebA(root=data_root, split="test",  transform=tfm, target_type="attr", download=download)

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
    return_ids: bool = False,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """
    Returns train_loader, val_loader, test_loader, dataset_info

    Guarantees:
    - Deterministic subset selection (optional) using subset_seed (defaults to seed)
    - Deterministic train/val split using seed
    - Deterministic worker seeding (numpy/random)
    - Optional stable, collision-proof sample IDs via return_ids

    Batch format:
    - return_ids=False (default): (x, y)
    - return_ids=True:            (x, y, sid)
      where sid = (dataset_name, split_name, root_idx)
    """
    base_train, base_test, info = _load_base_datasets(dataset_name, data_root, download=download)

    # Optional subset on the *train* pool (before train/val split)
    if subset_size is not None and subset_size < len(base_train):
        ss = seed if subset_seed is None else subset_seed
        rng = np.random.default_rng(ss)
        indices = rng.choice(len(base_train), size=subset_size, replace=False).tolist()
        base_train = Subset(base_train, indices)

    # Validate split
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1). Got {val_split}")

    n_total = len(base_train)
    n_val = int((val_split * n_total))
    n_val = max(1, min(n_val, n_total - 1))
    n_train = n_total - n_val

    train_view, val_view = random_split(
        base_train,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Wrap for subject IDS
    train_ds = IndexedDataset(train_view, dataset_name=info.name, split_name="train")
    val_ds   = IndexedDataset(val_view,   dataset_name=info.name, split_name="val")
    test_ds  = IndexedDataset(base_test,  dataset_name=info.name, split_name="test")

    # DataLoader generators
    g_train = torch.Generator().manual_seed(seed)
    g_eval  = torch.Generator().manual_seed(seed + 1)

    use_persistent = bool(persistent_workers and num_workers > 0)
    collate_fn = _make_collate_fn(return_ids=return_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g_train,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
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
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g_eval if shuffle_test else None,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, info