from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def load_full_config(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load training config JSON and split into structured sub-config dictionaries.

    Returns a dictionary with four sections:

    ---------------------------------------------------------------------------
    model: Dict[str, Any]
        Contains all parameters needed to reconstruct the model architecture.
        Keys:
            - base_channels: int
            - gn_groups: int
            - model_variant: str
            - num_levels: int
            - out_activation: str
            - z_dim: int
            - dataset: str

    ---------------------------------------------------------------------------
    data: Dict[str, Any]
        Contains dataset and dataloader configuration.
        Keys:
            - batch_size: int
            - data_root: str
            - download: bool
            - drop_last: bool
            - num_workers: int
            - persistent_workers: bool
            - pin_memory: bool
            - return_ids: bool
            - shuffle_train: bool
            - shuffle_val: bool
            - shuffle_test: bool
            - subset_seed: Optional[int]
            - subset_size: Optional[int]
            - val_split: float
            - dataset: str
            - seed: Optional[int]

    ---------------------------------------------------------------------------
    train: Dict[str, Any]
        Contains training loop hyperparameters.
        Keys:
            - epochs: int
            - lr: float
            - weight_decay: float
            - grad_clip: float
            - use_amp: bool
            - early_stop_patience: int
            - early_stop_min_delta: float
            - log_every: int
            - save_dir: str
            - loss: Optional[str]
            - device: Optional[str]
            - seed: Optional[int]

    ---------------------------------------------------------------------------
    meta: Dict[str, Any]
        Metadata useful for evaluation / logging.
        Keys:
            - run_name: Optional[str]
            - use_amp: Optional[bool]
            - seed: Optional[int]
            - device: Optional[str]
    ---------------------------------------------------------------------------

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required top-level keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = json.load(f)

    # --- basic validation ---
    required_keys = ["model", "data", "train", "dataset", "z_dim"]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing required key in config: '{k}'")

    # --- build structured configs ---

    model_cfg = {
        **cfg["model"],
        "z_dim": cfg["z_dim"],
        "dataset": cfg["dataset"],
    }

    data_cfg = {
        **cfg["data"],
        "dataset": cfg["dataset"],
        "seed": cfg.get("seed"),
    }

    train_cfg = {
        **cfg["train"],
        "loss": cfg.get("loss"),
        "device": cfg.get("device"),
        "seed": cfg.get("seed"),
    }

    meta_cfg = {
        "run_name": cfg.get("run_name"),
        "use_amp": cfg.get("use_amp"),
        "seed": cfg.get("seed"),
        "device": cfg.get("device"),
    }

    return {
        "model": model_cfg,
        "data": data_cfg,
        "train": train_cfg,
        "meta": meta_cfg,
    }