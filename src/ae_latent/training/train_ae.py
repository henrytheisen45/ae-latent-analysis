from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler


# ----------------------------
# Configs
# ----------------------------

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0  # 0 or None disables
    use_amp: bool = True

    # Early stopping
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0

    # Logging / saving
    log_every: int = 50
    save_dir: str = "runs"
    run_name: str = "ae_run"

    # Repro-ish
    seed: int = 42


# ----------------------------
# Helpers
# ----------------------------

def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility (slower). Uncomment if you want maximal determinism.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def unpack_batch(batch):
    """
    Supports:
      (x, y)
      (x, y, sid)
    Returns x and optionally sid.
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
        Returns True if should stop.
        """
        if self.best is None or (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[GradScaler], meta: Dict[str, Any]) -> None:
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

        with autocast(device_type="cuda", enabled=use_amp):
            recon = model(x)
            loss = loss_fn(recon, x)

        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


# ----------------------------
# Main training loop
# ----------------------------

def train_autoencoder(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    device: torch.device,
    cfg: TrainConfig,
    loss_name: str = "l1",  # "l1" or "mse"
) -> Dict[str, Any]:
    """
    Trains an autoencoder to reconstruct x.

    Returns a dict with training history + best checkpoint path.
    """
    set_torch_seed(cfg.seed)

    model = model.to(device)

    if loss_name == "l1":
        loss_fn = nn.L1Loss()
    elif loss_name == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    stopper = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)

    run_dir = Path(cfg.save_dir) / cfg.run_name
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "cfg": asdict(cfg),
        "loss_name": loss_name,
    }

    best_val = float("inf")

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_loader):
            x, _sid = unpack_batch(batch)
            x = x.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=cfg.use_amp and device.type == "cuda"):
                recon = model(x)
                loss = loss_fn(recon, x)

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            running += float(loss.item()) * bs
            seen += bs
            global_step += 1

            if cfg.log_every > 0 and (step % cfg.log_every == 0):
                print(f"epoch {epoch:03d} step {step:04d} loss {loss.item():.4f}")

        train_loss = running / max(1, seen)
        val_loss = evaluate_recon_loss(model, val_loader, device, loss_fn, use_amp=(cfg.use_amp and device.type == "cuda"))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[epoch {epoch:03d}] train {train_loss:.6f} | val {val_loss:.6f}")

        # Save last
        save_checkpoint(
            last_path, model, opt, scaler,
            meta={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                best_path, model, opt, scaler,
                meta={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": True},
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