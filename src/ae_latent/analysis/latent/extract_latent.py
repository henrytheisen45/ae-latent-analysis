from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def extract_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cuda",
    use_amp: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract latent vectors z = model.encode(x) over a dataloader.

    Works for:
        - Any z_dim
        - Any image size
        - Any dataset
        - Batches returning (x), (x,y), or (x,y,idx)

    Returns:
        z   : (N, z_dim) float32
        y   : (N,) int64 or None
        idx : (N,) int64 or None
    """

    if not hasattr(model, "encode"):
        raise AttributeError("Model must implement encode(x) -> z")

    device = torch.device(device)
    model.eval().to(device)

    zs = []
    ys = []
    idxs = []

    for b, batch in enumerate(loader):

        if max_batches is not None and b >= max_batches:
            break

        # Handle different batch formats
        y = idx = None

        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                x, y, idx = batch
            elif len(batch) == 2:
                x, y = batch
            elif len(batch) == 1:
                (x,) = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        else:
            x = batch

        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            z = model.encode(x)

        if z.ndim != 2:
            raise ValueError(
                f"encode(x) must return shape (B, z_dim). Got {tuple(z.shape)}"
            )

        zs.append(z.detach().float().cpu().numpy())

        if y is not None:
            ys.append(
                y.detach().cpu().numpy()
                if torch.is_tensor(y)
                else np.asarray(y)
            )

        if idx is not None:
            idxs.append(
                idx.detach().cpu().numpy()
                if torch.is_tensor(idx)
                else np.asarray(idx)
            )

    z_all = np.concatenate(zs, axis=0).astype(np.float32)

    y_all = np.concatenate(ys, axis=0).astype(np.int64) if ys else None
    idx_all = np.concatenate(idxs, axis=0).astype(np.int64) if idxs else None

    return z_all, y_all, idx_all


def save_latents(
    out_path: str | Path,
    z: np.ndarray,
    y: Optional[np.ndarray] = None,
    idx: Optional[np.ndarray] = None,
    **extra_arrays,
) -> Path:
    """
    Save latents to compressed .npz
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"z": z}
    if y is not None:
        payload["y"] = y
    if idx is not None:
        payload["idx"] = idx
    payload.update(extra_arrays)

    np.savez_compressed(out_path, **payload)
    return out_path