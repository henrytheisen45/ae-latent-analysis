# ae_latent/latents/extract.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


Sid = Tuple[str, str, int]  # (dataset_name, split_name, base_idx)


def extract_latents_to_npz(
    model: nn.Module,
    dataloader: DataLoader,
    out_path: str | Path,
    *,
    overwrite: bool = False,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    max_items: Optional[int] = None,
) -> Path:
    """
    Extract latent representations z for an entire dataloader and save to .npz.

    REQUIREMENTS:
      - dataloader must yield batches of (x, y, sid) where sid is:
          (dataset_name: str, split_name: str, base_idx: int)
        This matches your IndexedDataset + return_ids=True path.
      - model must implement either:
          - model.encode(x) -> z (preferred), OR
          - model.forward(x) returns a tuple containing z somewhere (not supported here by default)

    Saves an .npz with:
      - Z: float32 array shape (N, d)
      - y: int64 array shape (N, ...)
      - base_idx: int64 array shape (N,)
      - dataset: unicode array shape (N,) (same value repeated per sample)
      - split: unicode array shape (N,)   (same value repeated per sample)

    Args:
      overwrite: if False and out_path exists, raises.
      use_amp: use autocast on CUDA for faster inference (safe for encoding).
      amp_dtype: dtype for autocast (default float16).
      max_items: if set, stop after saving this many samples (useful for quick tests).

    Returns:
      Path to the saved npz.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    model.eval()
    device = _infer_model_device(model)

    if not hasattr(model, "encode") or not callable(getattr(model, "encode")):
        raise AttributeError("Model must implement model.encode(x) -> z for latent extraction.")

    Z_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    base_idx_list: list[int] = []
    dataset_list: list[str] = []
    split_list: list[str] = []

    total = 0

    # Autocast only makes sense on CUDA.
    use_autocast = bool(use_amp and device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_autocast else _nullcontext()

    with torch.inference_mode(), autocast_ctx:
        for batch in dataloader:
            x, y, sid = _unpack_batch_with_ids(batch)

            x = x.to(device, non_blocking=True)

            z = model.encode(x)
            if not isinstance(z, torch.Tensor):
                raise TypeError(f"model.encode(x) must return a torch.Tensor, got {type(z)}")
            if z.ndim != 2:
                raise ValueError(f"Expected latent z to have shape (B, d). Got shape {tuple(z.shape)}")

            # Move to CPU and convert to numpy
            z_np = z.detach().to("cpu").to(torch.float32).numpy()
            y_np = _to_numpy_int64(y)

            # Parse IDs
            ds_names, split_names, base_idxs = _parse_sid_batch(sid)

            # Accumulate
            Z_chunks.append(z_np)
            y_chunks.append(y_np)
            base_idx_list.extend(base_idxs)
            dataset_list.extend(ds_names)
            split_list.extend(split_names)

            total += z_np.shape[0]

            if max_items is not None and total >= max_items:
                # trim to exact max_items
                trim = total - max_items
                if trim > 0:
                    # trim last chunk
                    Z_chunks[-1] = Z_chunks[-1][:-trim]
                    y_chunks[-1] = y_chunks[-1][:-trim]
                    del base_idx_list[-trim:]
                    del dataset_list[-trim:]
                    del split_list[-trim:]
                total = max_items
                break

    if total == 0:
        raise ValueError("No samples were extracted (dataloader yielded nothing).")

    Z_all = np.concatenate(Z_chunks, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(y_chunks, axis=0).astype(np.int64, copy=False)

    base_idx_all = np.asarray(base_idx_list, dtype=np.int64)
    dataset_all = np.asarray(dataset_list, dtype=np.str_)
    split_all = np.asarray(split_list, dtype=np.str_)

    # Sanity checks
    n = Z_all.shape[0]
    if y_all.shape[0] != n or base_idx_all.shape[0] != n or dataset_all.shape[0] != n or split_all.shape[0] != n:
        raise RuntimeError("Internal length mismatch while assembling arrays.")

    np.savez_compressed(
        out_path,
        Z=Z_all,
        y=y_all,
        base_idx=base_idx_all,
        dataset=dataset_all,
        split=split_all,
    )

    return out_path


# ----------------------------
# Helpers
# ----------------------------

def _infer_model_device(model: nn.Module) -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        # no parameters? very unusual, but handle it
        return torch.device("cpu")


def _unpack_batch_with_ids(batch: Any):
    """
    Expect (x, y, sid). Raise loudly if not.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise ValueError(
            "Dataloader must yield (x, y, sid). "
            "Did you set return_ids=True and use the IndexedDataset wrapper?"
        )
    return batch[0], batch[1], batch[2]


def _parse_sid_batch(sid: Any) -> tuple[list[str], list[str], list[int]]:
    """
    sid comes from default_collate over a list of tuples like:
      ("celeba","train",123)

    Default collation turns that into a sequence of length 3:
      sid[0] = tuple[str] dataset names (len B)
      sid[1] = tuple[str] split names   (len B)
      sid[2] = tensor/int list base_idx (len B)

    We normalize to python lists.
    """
    if not isinstance(sid, (tuple, list)) or len(sid) != 3:
        raise ValueError(f"sid must be a (dataset_names, split_names, base_idxs) triple. Got type={type(sid)} value={sid}")

    ds_names = list(sid[0])
    split_names = list(sid[1])

    base = sid[2]
    if isinstance(base, torch.Tensor):
        base_idxs = base.detach().to("cpu").to(torch.int64).tolist()
    else:
        base_idxs = [int(b) for b in base]

    if not (len(ds_names) == len(split_names) == len(base_idxs)):
        raise ValueError("sid components have different lengths.")

    # Quick type validation (fail fast)
    for d in ds_names:
        if not isinstance(d, str):
            raise TypeError(f"dataset_name in sid must be str, got {type(d)}")
    for s in split_names:
        if not isinstance(s, str):
            raise TypeError(f"split_name in sid must be str, got {type(s)}")

    return ds_names, split_names, base_idxs


def _to_numpy_int64(y: Any) -> np.ndarray:
    """
    Convert y to an int64 numpy array, preserving shape (B,) or (B, K).
    Works for class labels and CelebA attribute vectors.
    """
    if isinstance(y, torch.Tensor):
        return y.detach().to("cpu").to(torch.int64).numpy()
    # If torchvision returns int, list, etc.
    return np.asarray(y, dtype=np.int64)


class _nullcontext:
    def __enter__(self):  # noqa: D401
        return None
    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False