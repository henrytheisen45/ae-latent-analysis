"""
train.py

Minimal, reproducible training CLI for vector-bottleneck autoencoders.

Philosophy:
- Hard-code sensible defaults for "first pass" experiments to get models for testing and building analysis

Assumes:
- get_dataloaders(...) returns train_loader, val_loader, test_loader, info
- Model file provides VectorLatentAE (and optionally GlobalAvgPoolLatentAE)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# ---- Change these imports to match your project structure ----
# from ae_latent_analysis.data.dataloaders import get_dataloaders
# from ae_latent_analysis.models.vect_latent_ae import VectorLatentAE, GlobalAvgPoolLatentAE, count_parameters

from ae_latent.data.loaders import get_dataloaders
from ae_latent.models.vect_latent_ae import VectorLatentAE, GlobalAvgPoolLatentAE, count_parameters  

# ----------------------------
# Defaults (future configs)
# ----------------------------

DEFAULT_DATA: Dict[str, Any] = dict(
    data_root="/data/users/theisehr9736/lsgeometry/unet_optuna/data",
    batch_size=128,
    num_workers=4,
    val_split=0.1,
    subset_size=None,
    subset_seed=None,
    return_ids=False, 
    shuffle_train=True,
    shuffle_val=False,
    shuffle_test=False,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    download=True,
)

DEFAULT_MODEL: Dict[str, Any] = dict(
    model_variant="vector", 
    base_channels=32,
    num_levels=4,
    gn_groups=8,
    out_activation="tanh",
)

DEFAULT_TRAIN: Dict[str, Any] = dict(
    epochs=100,
    lr=2e-4,
    weight_decay=1e-4,
    grad_clip=1.0,             # 0 disables
    use_amp=True,

    # early stopping
    early_stop_patience=10,
    early_stop_min_delta=0.0,

    # logging / saving
    log_every=50,
    save_dir="runs",
)


# ----------------------------
# Small utilities
# ----------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # If you need maximal determinism (slower), uncomment:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def device_from_args() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unpack_batch(batch):
    """
    Supports:
      (x, y)
      (x, y, sid)
    Returns x (and sid optionally; unused for training).
    """
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected batch as tuple/list, got {type(batch)}")

    if len(batch) == 2:
        x, _y = batch
        sid = None
    elif len(batch) == 3:
        x, _y, sid = batch
    else:
        raise ValueError(f"Unexpected batch len={len(batch)}; expected 2 or 3")
    return x, sid


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best is None or (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "meta": meta,
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate_recon_loss(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0
    n = 0

    for batch in loader:
        x, _sid = unpack_batch(batch)
        x = x.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
            recon = model(x)
            loss = loss_fn(recon, x)

        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


# ----------------------------
# Main training
# ----------------------------

def build_model(
    *,
    variant: str,
    in_channels: int,
    img_shape: Tuple[int, int],
    z_dim: int,
    base_channels: int,
    num_levels: int,
    out_activation: str,
    gn_groups: int,
) -> nn.Module:
    v = variant.lower().strip()
    if v == "vector":
        return VectorLatentAE(
            in_channels=in_channels,
            img_shape=img_shape,
            z_dim=z_dim,
            base_channels=base_channels,
            num_levels=num_levels,
            out_activation=out_activation,  # type: ignore[arg-type]
            gn_groups=gn_groups,
        )
    if v == "gap":
        return GlobalAvgPoolLatentAE(
            in_channels=in_channels,
            img_shape=img_shape,
            z_dim=z_dim,
            base_channels=base_channels,
            num_levels=num_levels,
            out_activation=out_activation,  # type: ignore[arg-type]
            gn_groups=gn_groups,
        )
    raise ValueError(f"Unknown model_variant={variant!r}. Use 'vector' or 'gap'.")


def train(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    use_amp: bool,
    early_stop_patience: int,
    early_stop_min_delta: float,
    log_every: int,
    run_dir: Path,
    loss_name: str,
    seed: int,
) -> Dict[str, Any]:
    set_torch_seed(seed)
    model = model.to(device)

    if loss_name == "l1":
        loss_fn = nn.L1Loss()
    elif loss_name == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name!r}. Use 'l1' or 'mse'.")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    stopper = EarlyStopping(early_stop_patience, early_stop_min_delta)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_loader):
            x, _sid = unpack_batch(batch)
            x = x.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                recon = model(x)
                loss = loss_fn(recon, x)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            running += float(loss.item()) * bs
            seen += bs
            global_step += 1

            if log_every > 0 and (step % log_every == 0):
                print(f"epoch {epoch:03d} step {step:04d} loss {loss.item():.6f}")

        train_loss = running / max(1, seen)
        val_loss = evaluate_recon_loss(model, val_loader, device, loss_fn, use_amp=use_amp)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[epoch {epoch:03d}] train {train_loss:.6f} | val {val_loss:.6f}")

        # Save last
        save_checkpoint(
            last_path, model, opt, scaler,
            meta={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "global_step": global_step},
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                best_path, model, opt, scaler,
                meta={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": True, "global_step": global_step},
            )
            print(f"  -> new best saved: {best_path}")

        # Early stop
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch} (best val={stopper.best:.6f}).")
            break

    return {
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_val": best_val,
        "history": history,
    }


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vector-bottleneck autoencoder (minimal CLI).")

    # Only the knobs you actually sweep / vary right now
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset name: mnist, fashion_mnist, cifar10, cifar100, svhn, celeba")
    p.add_argument("--z-dim", type=int, required=True, help="Latent dimension z_dim")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run name. If omitted, auto-generated from dataset/z_dim/time.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # One optional switch kept for future but defaulted sanely
    p.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"], help="Reconstruction loss")

    # A safety valve: disable AMP if it ever causes issues
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_args()

    # Resolve configs (hard-coded defaults + minimal overrides)
    data_cfg = dict(DEFAULT_DATA)
    model_cfg = dict(DEFAULT_MODEL)
    train_cfg = dict(DEFAULT_TRAIN)

    # Overrides
    seed = int(args.seed)
    z_dim = int(args.z_dim)
    dataset = str(args.dataset)
    loss_name = str(args.loss)
    use_amp = bool(train_cfg["use_amp"] and (not args.no_amp))

    run_name = args.run_name
    if run_name is None or run_name.strip() == "":
        run_name = f"{dataset}_z{z_dim}_{now_tag()}"

    run_dir = Path(train_cfg["save_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    resolved = {
        "dataset": dataset,
        "z_dim": z_dim,
        "seed": seed,
        "device": str(device),
        "loss": loss_name,
        "use_amp": use_amp,
        "run_name": run_name,
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, sort_keys=True)

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")

    # Build data
    train_loader, val_loader, test_loader, info = get_dataloaders(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        data_root=data_cfg["data_root"],
        val_split=data_cfg["val_split"],
        seed=seed,
        subset_size=data_cfg["subset_size"],
        subset_seed=data_cfg["subset_seed"],
        return_ids=data_cfg["return_ids"],
        shuffle_train=data_cfg["shuffle_train"],
        shuffle_val=data_cfg["shuffle_val"],
        shuffle_test=data_cfg["shuffle_test"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=data_cfg["persistent_workers"],
        drop_last=data_cfg["drop_last"],
        download=data_cfg["download"],
    )

    img_shape = (info.img_size, info.img_size)

    # Build model
    model = build_model(
        variant=model_cfg["model_variant"],
        in_channels=info.channels,
        img_shape=img_shape,
        z_dim=z_dim,
        base_channels=model_cfg["base_channels"],
        num_levels=model_cfg["num_levels"],
        out_activation=model_cfg["out_activation"],
        gn_groups=model_cfg["gn_groups"],
    )

    n_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Params: {n_params:,}")
    print(f"Input shape: (C,H,W)=({info.channels},{info.img_size},{info.img_size})")
    print(f"Bottom shape: {getattr(model, 'bottom_shape', None)} | flat_dim: {getattr(model, 'flat_dim', None)}")

    # Train
    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        grad_clip=float(train_cfg["grad_clip"]),
        use_amp=use_amp,
        early_stop_patience=int(train_cfg["early_stop_patience"]),
        early_stop_min_delta=float(train_cfg["early_stop_min_delta"]),
        log_every=int(train_cfg["log_every"]),
        run_dir=run_dir,
        loss_name=loss_name,
        seed=seed,
    )

    # Final eval on test (optional but useful)
    if loss_name == "l1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()
    test_loss = evaluate_recon_loss(model.to(device), test_loader, device, loss_fn, use_amp=use_amp)
    result["test_loss"] = test_loss

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"Done. Best val: {result['best_val']:.6f} | test: {test_loss:.6f}")
    print(f"Best ckpt: {result['best_path']}")
    print(f"Last ckpt: {result['last_path']}")


if __name__ == "__main__":
    main()