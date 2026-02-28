from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from ae_latent.utils.config import load_yaml, prepare_run
from ae_latent.data.loaders import build_dataloaders


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Minimal stub. Replace with real factory.
    Expected config fields:
      cfg["model"]["name"] etc.
    """
    name = str(cfg["model"]["name"])
    if name != "VectorLatentAE":
        raise ValueError(f"Unknown model name: {name}. Implement build_model() factory.")
    # TODO: replace with  actual import / constructor:
    # from ae_latent.models.vector_latent_ae import VectorLatentAE
    # return VectorLatentAE(**cfg["model"], out_activation=...)
    raise NotImplementedError("Implement model factory in cli/train.py or in ae_latent.models.factory")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an autoencoder from a YAML spec.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML experiment spec.")
    ap.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda, cuda:0, cpu")
    args = ap.parse_args()

    spec_path = Path(args.config)
    spec = load_yaml(spec_path)

    # Resolve config + create unique run directory + write spec/resolved artifacts
    run_dir, cfg = prepare_run(spec=spec, spec_path=spec_path)
    print(f"[train] run_dir = {run_dir}")

    # Dataloaders (uses cfg['data']['normalize'] and cfg['data']['val_split_seed'])
    loaders = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Model
    model = build_model(cfg)

    # Training loop
    # TODO: replace with actual training function
    # from ae_latent.training.train_loop import train_autoencoder
    # out = train_autoencoder(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     cfg=...,  # convert cfg["train"] + cfg["loss"] into TrainConfig dataclass
    #     device=args.device,
    # )
    # print(out)

    raise NotImplementedError("Next: wire in training loop and TrainConfig mapping.")


if __name__ == "__main__":
    main()