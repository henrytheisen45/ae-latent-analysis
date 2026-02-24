from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def build_vect_latent_ae(
    model_cfg: Dict[str, Any],
    *,
    in_channels: Optional[int] = None,
    img_shape: Optional[Tuple[int, int]] = None,
    device: str = "cpu",
    ckpt_path: Optional[str | Path] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Build the vector-latent autoencoder (VectorLatentAE) from config.

    Required model_cfg keys:
      - z_dim: int
      - base_channels: int
      - num_levels: int
      - out_activation: str  ("tanh" expected if inputs normalized to [-1,1])
    Optional:
      - gn_groups: int

    You SHOULD pass in_channels and img_shape explicitly (best).
    If you don't, this function will try to infer them from model_cfg["dataset"].
    """
    # ---- required keys ----
    for k in ("z_dim", "base_channels", "num_levels"):
        if k not in model_cfg:
            raise ValueError(f"model_cfg missing required key: '{k}'")

    z_dim = int(model_cfg["z_dim"])
    base_channels = int(model_cfg["base_channels"])
    num_levels = int(model_cfg["num_levels"])
    gn_groups = int(model_cfg.get("gn_groups", 8))
    out_activation = str(model_cfg.get("out_activation", "tanh"))

    # ---- avoid silent wrong shapes ----
    if in_channels is None or img_shape is None:
        ds = model_cfg.get("dataset")
        if ds is None:
            raise ValueError(
                "Must provide in_channels and img_shape, or set model_cfg['dataset'] "
                "so they can be inferred."
            )
        in_channels, img_shape = _infer_dataset_io(ds)

    assert in_channels is not None
    assert img_shape is not None

    # ---- construct ----
    from ae_latent.models.vect_latent_ae import VectorLatentAE 

    model = VectorLatentAE(
        in_channels=in_channels,
        img_shape=img_shape,
        z_dim=z_dim,
        base_channels=base_channels,
        num_levels=num_levels,
        gn_groups=gn_groups,
        out_activation=out_activation,
    ).to(device)

    # ---- optional checkpoint load ----
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _strip_state_dict_prefixes(state, prefixes=("model.", "net.", "module."))

        missing, unexpected = model.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            # In strict mode, PyTorch would already have raised, but keep this for clarity.
            raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

    return model


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Handle common checkpoint formats."""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # maybe it's already a raw state dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt  # type: ignore[return-value]
    raise ValueError("Unrecognized checkpoint format; couldn't find a state_dict.")


def _strip_state_dict_prefixes(
    state: Dict[str, torch.Tensor],
    prefixes: Tuple[str, ...],
) -> Dict[str, torch.Tensor]:
    """Strip a single matching prefix from all keys if it looks consistent."""
    keys = list(state.keys())
    for p in prefixes:
        if keys and all(k.startswith(p) for k in keys):
            return {k[len(p):]: v for k, v in state.items()}
    return state


def _infer_dataset_io(dataset: str) -> tuple[int, Tuple[int, int]]:
    """
    Temporary inference. Prefer explicit config instead.
    Adjust to match YOUR training transforms.
    """
    ds = dataset.lower()
    if ds in {"mnist", "fashionmnist"}:
        return 1, (28, 28)
    if ds == "cifar10":
        return 3, (32, 32)
    if ds == "celeba":
        # Only correct if you trained on 64x64. If not, this is wrong.
        return 3, (64, 64)
    raise ValueError(f"Unknown dataset '{dataset}' for inference.")