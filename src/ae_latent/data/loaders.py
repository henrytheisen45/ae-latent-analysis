from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

SplitName = Literal["train", "val", "test"]
NormalizeMode = Literal["-1_1", "0_1"]


# ----------------------------
# Determinism helpers
# ----------------------------

def seed_worker(worker_id: int) -> None:
    """
    Seed numpy/random deterministically inside each DataLoader worker.

    PyTorch assigns each worker a unique base seed; we derive numpy/random from it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed) 


def root_index(ds: Dataset, idx: int) -> int:
    """
    Map an index in an arbitrarily nested Subset(...) view back to the root dataset index.

    If ds is not a Subset, this returns idx (the index into that dataset object).
    """
    cur = ds
    i = int(idx)
    while isinstance(cur, Subset):
        i = int(cur.indices[i])
        cur = cur.dataset
    return int(i)


# ----------------------------
# Dataset wrapper that returns IDs
# ----------------------------

class WithSampleID(Dataset):
    """
    Wrap a dataset view so __getitem__ returns (x, y, sid),
    where sid = (dataset_name, split_name, root_idx).

    IMPORTANT meaning of root_idx:
    - root_idx is the index in the underlying *root dataset object for that split*.
      * For train/val, root_idx refers to indices in the "train_full" pool dataset.
      * For test, root_idx refers to indices in the test dataset object.
    - This is stable under Subset nesting, and collision-proof across splits because
      split_name is included in sid.
    """
    def __init__(self, view: Dataset, *, dataset_name: str, split_name: SplitName):
        self.view = view
        self.dataset_name = str(dataset_name)
        self.split_name: SplitName = split_name

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, idx: int):
        item = self.view[idx]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Dataset must return at least (x, y).")
        x, y = item[0], item[1]

        ridx = root_index(self.view, idx)
        sid = (self.dataset_name, self.split_name, ridx)
        return x, y, sid


# ----------------------------
# DatasetInfo
# ----------------------------

@dataclass(frozen=True)
class DatasetInfo:
    name: str
    channels: int
    target_kind: Literal["class", "attrs"]
    target_dim: Optional[int]  # None for class datasets


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    name = dataset_name.lower()
    if name in {"cifar10", "cifar100", "svhn"}:
        return DatasetInfo(name=name, channels=3, target_kind="class", target_dim=None)
    if name == "celeba":
        return DatasetInfo(name=name, channels=3, target_kind="attrs", target_dim=40)
    raise ValueError(
        f"Unknown dataset: {dataset_name}. Expected one of: cifar10, cifar100, svhn, celeba"
    )


# ----------------------------
# Transforms + dataset loading
# ----------------------------

def _parse_normalize_mode(v: Any) -> NormalizeMode:
    if v is None:
        return "-1_1"
    if not isinstance(v, str):
        raise ValueError(f"cfg['data']['normalize'] must be a string, got {type(v)}")
    s = v.strip()
    if s not in ("-1_1", "0_1"):
        raise ValueError(f"cfg['data']['normalize'] must be '-1_1' or '0_1', got {v!r}")
    return s  # type: ignore[return-value]


def make_transform(
    *,
    img_shape: Tuple[int, int],
    channels: int,
    normalize: NormalizeMode = "-1_1",
) -> transforms.Compose:
    """
    Deterministic preprocessing:
      - Resize to model img_shape
      - ToTensor
      - Normalize:
          "-1_1": map [0,1] -> [-1,1] (matches out_activation='tanh')
          "0_1":  keep [0,1] as-is
    """
    h, w = int(img_shape[0]), int(img_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid img_shape: {img_shape}")

    ops = [
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ]

    if normalize == "-1_1":
        if channels == 1:
            ops.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif normalize == "0_1":
        # Leave as [0,1]
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    return transforms.Compose(ops)


def load_datasets(
    *,
    dataset_name: str,
    root: str,
    download: bool,
    transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """
    Returns (train_full, test).

    train_full is the pool we split into train/val deterministically.
    test is the test split dataset object (separate from train_full for these torchvision datasets).
    """
    name = dataset_name.lower()

    if name == "cifar10":
        train_full = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        test = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        return train_full, test

    if name == "cifar100":
        train_full = datasets.CIFAR100(root=root, train=True, download=download, transform=transform)
        test = datasets.CIFAR100(root=root, train=False, download=download, transform=transform)
        return train_full, test

    if name == "svhn":
        train_full = datasets.SVHN(root=root, split="train", download=download, transform=transform)
        test = datasets.SVHN(root=root, split="test", download=download, transform=transform)
        return train_full, test

    if name == "celeba":
        # Explicitly enforce attributes as targets.
        train_full = datasets.CelebA(
            root=root,
            split="train",
            download=download,
            transform=transform,
            target_type="attr",
        )
        test = datasets.CelebA(
            root=root,
            split="test",
            download=download,
            transform=transform,
            target_type="attr",
        )
        return train_full, test

    raise ValueError(f"Dataset not implemented: {dataset_name}")


# ----------------------------
# Split persistence
# ----------------------------

def _split_path(save_dir: str, dataset_name: str, val_split: float, seed: int) -> Path:
    """
    Store split indices under:
      {save_dir}/splits/{dataset}_val{val_split}_seed{seed}.json

    Including val_split + seed avoids collisions across experiments.
    """
    safe_vs = str(val_split).replace(".", "p")
    return Path(save_dir) / "splits" / f"{dataset_name}_val{safe_vs}_seed{seed}.json"


def load_or_create_split_indices(
    *,
    n: int,
    dataset_name: str,
    val_split: float,
    seed: int,
    save_dir: str,
) -> Dict[str, Sequence[int]]:
    """
    Create or load deterministic train/val indices for the train_full pool dataset.
    """
    if not (0.0 <= val_split < 1.0):
        raise ValueError(f"val_split must be in [0,1). Got {val_split}")
    if n <= 1:
        raise ValueError(f"Dataset too small to split: n={n}")

    path = _split_path(save_dir, dataset_name, val_split, seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "train" not in data or "val" not in data:
            raise ValueError(f"Malformed split file: {path}")
        return data

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    n_val = int(round(n * float(val_split)))
    n_val = max(1, min(n_val, n - 1))

    train_idx = perm[:-n_val]
    val_idx = perm[-n_val:]

    data = {"train": train_idx, "val": val_idx}
    path.write_text(json.dumps(data, indent=2))
    return data


# ----------------------------
# Public entrypoint: config-driven builder
# ----------------------------

def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build dataloaders from your YAML-parsed config dict.

    Required:
      cfg["run"]["seed"]
      cfg["run"]["save_dir"]
      cfg["model"]["img_shape"]  (used for transforms)
      cfg["data"]["dataset"]
      cfg["data"]["root"]
      cfg["data"]["batch_size"]

    Optional (with defaults):
      cfg["data"]["download"]            default True
      cfg["data"]["normalize"]           default "-1_1"
      cfg["data"]["num_workers"]         default 0
      cfg["data"]["pin_memory"]          default True
      cfg["data"]["persistent_workers"]  default True
      cfg["data"]["shuffle_train"]       default True
      cfg["data"]["drop_last"]           default False
      cfg["data"]["val_split"]           default 0.1
      cfg["data"]["val_split_seed"]      default cfg["run"]["seed"]

    Returns:
      {"train": DataLoader, "val": DataLoader, "test": DataLoader, "info": DatasetInfo}
    """
    if "run" not in cfg or not isinstance(cfg["run"], dict):
        raise ValueError("cfg must contain dict key: run")
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        raise ValueError("cfg must contain dict key: data")
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        raise ValueError("cfg must contain dict key: model")

    run = cfg["run"]
    data = cfg["data"]
    model = cfg["model"]

    seed = int(run["seed"])
    save_dir = str(run["save_dir"])

    dataset_name = str(data["dataset"]).lower()
    root = str(data["root"])
    download = bool(data.get("download", True))

    normalize_mode = _parse_normalize_mode(data.get("normalize", "-1_1"))

    batch_size = int(data["batch_size"])
    num_workers = int(data.get("num_workers", 0))
    pin_memory = bool(data.get("pin_memory", True))
    persistent_workers = bool(data.get("persistent_workers", True))
    shuffle_train = bool(data.get("shuffle_train", True))
    drop_last = bool(data.get("drop_last", False))
    val_split = float(data.get("val_split", 0.1))

    # NEW: split seed is separate from run.seed (defaults to run.seed)
    val_split_seed = int(data.get("val_split_seed", seed))

    img_shape_raw = model.get("img_shape", None)
    if not isinstance(img_shape_raw, (list, tuple)) or len(img_shape_raw) != 2:
        raise ValueError("cfg['model']['img_shape'] must be a list/tuple of length 2, e.g. [64, 64]")
    img_shape = (int(img_shape_raw[0]), int(img_shape_raw[1]))

    info = get_dataset_info(dataset_name)
    tfm = make_transform(img_shape=img_shape, channels=info.channels, normalize=normalize_mode)

    train_full, test_base = load_datasets(
        dataset_name=dataset_name,
        root=root,
        download=download,
        transform=tfm,
    )

    splits = load_or_create_split_indices(
        n=len(train_full),
        dataset_name=dataset_name,
        val_split=val_split,
        seed=val_split_seed,
        save_dir=save_dir,
    )

    train_view = Subset(train_full, splits["train"])
    val_view = Subset(train_full, splits["val"])

    # Always return IDs (x, y, sid)
    train_ds = WithSampleID(train_view, dataset_name=dataset_name, split_name="train")
    val_ds = WithSampleID(val_view, dataset_name=dataset_name, split_name="val")
    test_ds = WithSampleID(test_base, dataset_name=dataset_name, split_name="test")

    # Deterministic shuffling for train (still uses run.seed)
    g_train = torch.Generator().manual_seed(seed)

    worker_init = seed_worker if num_workers > 0 else None
    use_persistent = bool(persistent_workers and num_workers > 0)

    # IMPORTANT: don't pass prefetch_factor=None
    common_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        persistent_workers=use_persistent,
    )
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle_train,
        generator=g_train if shuffle_train else None,
        drop_last=drop_last,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader, "info": info}