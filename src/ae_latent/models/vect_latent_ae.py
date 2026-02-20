"""
vect_latent_ae.py

Vector-bottleneck convolutional autoencoder for latent-geometry experiments.

This module provides two *encoder-reduction* variants that share the exact same
convolutional encoder/decoder body to prevent copy/paste drift:

Variants
--------
1) VectorLatentAE (baseline)
   - Encoder reduction: FLATTEN the bottom feature map h_bottom
     (B, Cb, Hb, Wb) -> (B, Cb*Hb*Wb) before projecting to z.

2) GlobalAvgPoolLatentAE (ablation)
   - Encoder reduction: GLOBAL AVERAGE POOL (GAP) the bottom feature map h_bottom
     (B, Cb, Hb, Wb) -> (B, Cb) before projecting to z.

What differs (and what does NOT)
-------------------------------
- Shared (identical across variants):
  * Convolutional encoder that maps x -> h_bottom
  * Convolutional decoder that maps h_bottom -> x_logits (pre-activation)
  * Decoder structure/capacity is identical
  * Decoder head 'from_z' is identical (z_dim -> flat_dim) across variants

- Different (by design):
  * The encoder reduction (flatten vs GAP)
  * The input dimension (and thus parameter count) of the encoder projection head 'to_z' differs
    substantially (flat_dim -> z_dim vs c_bottom -> z_dim). This is an unavoidable consequence of
    these reductions if you keep the body identical. Interpret comparisons accordingly.

Shape contract
--------------
- img_shape=(H, W) must be divisible by 2**num_levels in both dimensions.
  If not, pad/crop in the dataloader (recommended) rather than hacking the model.

Input / output scaling
----------------------
- If inputs are normalized to [-1, 1] (e.g., Normalize(mean=0.5,std=0.5)), use out_activation="tanh".
- If inputs are scaled to [0, 1], use out_activation="sigmoid".
- If you want raw logits (e.g., custom loss), use out_activation="none".

Design goals
------------
- Deterministic shapes and explicit contracts
- Shared core to avoid drift between variants
- Strong validation checks with clear error messages
- Stable defaults for small batches (GroupNorm + SiLU)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn

OutAct = Literal["tanh", "sigmoid", "none"]


# ----------------------------
# Helpers
# ----------------------------

def _pick_gn_groups(num_channels: int, preferred: int = 8) -> int:
    """
    GroupNorm requires num_groups to divide num_channels.
    Pick the largest g <= preferred that divides num_channels.
    Fallback to 1 (LayerNorm-like behavior across channels).
    """
    if not isinstance(num_channels, int) or num_channels <= 0:
        raise ValueError(f"num_channels must be a positive int, got {num_channels!r}")
    if not isinstance(preferred, int) or preferred <= 0:
        raise ValueError(f"preferred must be a positive int, got {preferred!r}")

    g = min(preferred, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _validate_nonneg_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")


class ConvBlock(nn.Module):
    """Conv -> GN -> SiLU -> Conv -> GN -> SiLU"""
    def __init__(self, in_ch: int, out_ch: int, gn_preferred: int = 8):
        super().__init__()
        _validate_positive_int("in_ch", in_ch)
        _validate_positive_int("out_ch", out_ch)
        g = _pick_gn_groups(out_ch, gn_preferred)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Downsample(nn.Module):
    """Downsample by 2 via stride-2 conv (kernel=4,stride=2,pad=1)."""
    def __init__(self, in_ch: int, out_ch: int, gn_preferred: int = 8):
        super().__init__()
        _validate_positive_int("in_ch", in_ch)
        _validate_positive_int("out_ch", out_ch)
        g = _pick_gn_groups(out_ch, gn_preferred)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Upsample(nn.Module):
    """Upsample by 2 via transposed conv (kernel=4,stride=2,pad=1)."""
    def __init__(self, in_ch: int, out_ch: int, gn_preferred: int = 8):
        super().__init__()
        _validate_positive_int("in_ch", in_ch)
        _validate_positive_int("out_ch", out_ch)
        g = _pick_gn_groups(out_ch, gn_preferred)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class _ShapeSpec:
    H: int
    W: int
    div: int
    Hb: int
    Wb: int
    c_bottom: int
    flat_dim: int


def _validate_img_shape(img_shape: Tuple[int, int]) -> Tuple[int, int]:
    if not (isinstance(img_shape, tuple) and len(img_shape) == 2):
        raise TypeError(f"img_shape must be a tuple (H, W), got {img_shape!r}")
    H, W = img_shape
    if not (isinstance(H, int) and isinstance(W, int)):
        raise TypeError(f"img_shape must contain ints, got {img_shape!r}")
    if H <= 0 or W <= 0:
        raise ValueError(f"img_shape must be positive, got {img_shape!r}")
    return H, W


def _validate_and_compute_shapes(img_shape: Tuple[int, int], base_channels: int, num_levels: int) -> _ShapeSpec:
    H, W = _validate_img_shape(img_shape)
    _validate_positive_int("base_channels", base_channels)
    _validate_nonneg_int("num_levels", num_levels)

    div = 2 ** num_levels
    if (H % div) != 0 or (W % div) != 0:
        raise ValueError(
            f"img_shape={img_shape} must be divisible by 2**num_levels={div} "
            f"(num_levels={num_levels}). Pad/crop in the dataloader."
        )

    Hb = H // div
    Wb = W // div
    c_bottom = base_channels * (2 ** num_levels)
    flat_dim = c_bottom * Hb * Wb
    return _ShapeSpec(H=H, W=W, div=div, Hb=Hb, Wb=Wb, c_bottom=c_bottom, flat_dim=flat_dim)


def _validate_out_activation(out_activation: OutAct) -> None:
    if out_activation not in ("tanh", "sigmoid", "none"):
        raise ValueError(f"out_activation must be one of ['tanh','sigmoid','none'], got {out_activation!r}")


# ----------------------------
# Shared core: encoder + decoder (no bottleneck choice here)
# ----------------------------

class _EncoderDecoderCore(nn.Module):
    """
    Shared convolutional core.

    Maps:
      x -> h_bottom where h_bottom has shape (B, c_bottom, Hb, Wb)
      h_bottom -> x_logits with shape (B, in_channels, H, W) (pre-output-activation)
    """
    def __init__(
        self,
        in_channels: int,
        img_shape: Tuple[int, int],
        base_channels: int,
        num_levels: int,
        gn_groups: int,
    ):
        super().__init__()
        _validate_positive_int("in_channels", in_channels)
        _validate_positive_int("gn_groups", gn_groups)

        spec = _validate_and_compute_shapes(img_shape, base_channels, num_levels)

        self.in_channels = in_channels
        self.img_shape = (spec.H, spec.W)
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.gn_groups = gn_groups

        # Exposed helpers for debugging/analysis
        self.bottom_shape: Tuple[int, int, int] = (spec.c_bottom, spec.Hb, spec.Wb)
        self.flat_dim: int = spec.flat_dim

        # Encoder: x -> h_bottom
        ch0 = base_channels
        enc = [ConvBlock(in_channels, ch0, gn_preferred=gn_groups)]
        c_in = ch0
        for level in range(num_levels):
            c_out = base_channels * (2 ** (level + 1))
            enc.append(Downsample(c_in, c_out, gn_preferred=gn_groups))
            enc.append(ConvBlock(c_out, c_out, gn_preferred=gn_groups))
            c_in = c_out
        self.encoder = nn.Sequential(*enc)

        # Decoder: h_bottom -> x_logits
        dec = [ConvBlock(spec.c_bottom, spec.c_bottom, gn_preferred=gn_groups)]
        c_in = spec.c_bottom
        for level in range(num_levels - 1, -1, -1):
            c_out = base_channels * (2 ** level)
            dec.append(Upsample(c_in, c_out, gn_preferred=gn_groups))
            dec.append(ConvBlock(c_out, c_out, gn_preferred=gn_groups))
            c_in = c_out
        self.decoder = nn.Sequential(*dec)
        self.out_conv = nn.Conv2d(ch0, in_channels, kernel_size=1, bias=True)

    # ---- validation helpers ----
    def check_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Expected x as 4D (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")
        if (x.shape[2], x.shape[3]) != self.img_shape:
            raise ValueError(f"Expected spatial {self.img_shape}, got {(x.shape[2], x.shape[3])}")

    def check_bottom(self, h: torch.Tensor) -> None:
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"Expected h to be a torch.Tensor, got {type(h)}")
        if h.ndim != 4:
            raise ValueError(f"Expected h as 4D (B,Cb,Hb,Wb), got {tuple(h.shape)}")
        cb, hb, wb = self.bottom_shape
        if h.shape[1] != cb:
            raise ValueError(f"Expected h channels Cb={cb}, got {h.shape[1]}")
        if (h.shape[2], h.shape[3]) != (hb, wb):
            raise ValueError(f"Expected h spatial {(hb, wb)}, got {(h.shape[2], h.shape[3])}")

    # ---- core transforms ----
    def encode_to_bottom(self, x: torch.Tensor) -> torch.Tensor:
        self.check_input(x)
        return self.encoder(x)

    def decode_from_bottom(self, h: torch.Tensor) -> torch.Tensor:
        self.check_bottom(h)
        x = self.decoder(h)
        x = self.out_conv(x)
        return x  # logits / pre-activation


# ----------------------------
# Base wrapper with shared forward + activation + get_latent_vector
# ----------------------------

class _BaseLatentAE(nn.Module):
    """
    Base AE wrapper.

    Subclasses must implement:
      - encode(x) -> z  (B, z_dim)
      - decode(z) -> x_recon (B, C, H, W)

    This base provides:
      - forward(x, return_latent)
      - output activation handling
      - get_latent_vector() convenience for analysis
    """
    def __init__(self, *, out_activation: OutAct):
        super().__init__()
        _validate_out_activation(out_activation)
        self.out_activation: OutAct = out_activation

    def _apply_out_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_activation == "tanh":
            return torch.tanh(x)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(x)
        return x  # "none"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement encode(x) -> z")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement decode(z) -> recon")

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        recon = self.decode(z)
        return (recon, z) if return_latent else recon

    @torch.no_grad()
    def get_latent_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience for analysis: returns detached z (B, z_dim).
        Preserves the module's training/eval mode.
        """
        was_training = self.training
        try:
            self.eval()
            return self.encode(x).detach()
        finally:
            self.train(was_training)


# ----------------------------
# Variant 1: FLATTEN bottleneck (baseline)
# ----------------------------

class VectorLatentAE(_BaseLatentAE):
    """
    Baseline model.

    Encoder reduction:
      h_bottom (B, Cb, Hb, Wb) -> flatten -> (B, Cb*Hb*Wb) -> Linear -> z (B, z_dim)
    """
    def __init__(
        self,
        in_channels: int,
        img_shape: Tuple[int, int],
        z_dim: int,
        base_channels: int = 32,
        num_levels: int = 4,
        out_activation: OutAct = "tanh",
        gn_groups: int = 8,
    ):
        # Public validation (fail early with clear errors)
        _validate_positive_int("in_channels", in_channels)
        _validate_img_shape(img_shape)
        _validate_positive_int("z_dim", z_dim)
        _validate_positive_int("base_channels", base_channels)
        _validate_nonneg_int("num_levels", num_levels)
        _validate_positive_int("gn_groups", gn_groups)
        _validate_out_activation(out_activation)

        super().__init__(out_activation=out_activation)

        self.z_dim = z_dim
        self.core = _EncoderDecoderCore(
            in_channels=in_channels,
            img_shape=img_shape,
            base_channels=base_channels,
            num_levels=num_levels,
            gn_groups=gn_groups,
        )

        # Expose debug helpers
        self.bottom_shape = self.core.bottom_shape
        self.flat_dim = self.core.flat_dim

        # Heads
        self.to_z = nn.Linear(self.core.flat_dim, z_dim)
        self.from_z = nn.Linear(z_dim, self.core.flat_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.core.encode_to_bottom(x)
        z = self.to_z(h.flatten(1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"decode expects z as torch.Tensor, got {type(z)}")
        if z.ndim != 2:
            raise ValueError(f"decode expects z shape (B, z_dim), got {tuple(z.shape)}")
        if z.shape[1] != self.z_dim:
            raise ValueError(f"decode expects z_dim={self.z_dim}, got {z.shape[1]}")

        B = z.shape[0]
        c_bottom, Hb, Wb = self.core.bottom_shape
        h = self.from_z(z).view(B, c_bottom, Hb, Wb)
        x_logits = self.core.decode_from_bottom(h)
        return self._apply_out_activation(x_logits)


# ----------------------------
# Variant 2: GAP bottleneck (ablation)
# ----------------------------

class GlobalAvgPoolLatentAE(_BaseLatentAE):
    """
    Ablation model.

    Encoder reduction:
      h_bottom (B, Cb, Hb, Wb) -> GAP -> (B, Cb) -> Linear -> z (B, z_dim)

    Note:
      Decoder remains identical to the baseline. The model still decodes from a
      learned dense spatial tensor produced by from_z: z -> flat_dim.
    """
    def __init__(
        self,
        in_channels: int,
        img_shape: Tuple[int, int],
        z_dim: int,
        base_channels: int = 32,
        num_levels: int = 4,
        out_activation: OutAct = "tanh",
        gn_groups: int = 8,
    ):
        # Public validation (fail early with clear errors)
        _validate_positive_int("in_channels", in_channels)
        _validate_img_shape(img_shape)
        _validate_positive_int("z_dim", z_dim)
        _validate_positive_int("base_channels", base_channels)
        _validate_nonneg_int("num_levels", num_levels)
        _validate_positive_int("gn_groups", gn_groups)
        _validate_out_activation(out_activation)

        super().__init__(out_activation=out_activation)

        self.z_dim = z_dim
        self.core = _EncoderDecoderCore(
            in_channels=in_channels,
            img_shape=img_shape,
            base_channels=base_channels,
            num_levels=num_levels,
            gn_groups=gn_groups,
        )

        # Expose debug helpers
        self.bottom_shape = self.core.bottom_shape
        self.flat_dim = self.core.flat_dim

        # Pool + heads
        self._pool = nn.AdaptiveAvgPool2d((1, 1))
        c_bottom, _, _ = self.core.bottom_shape
        self.to_z = nn.Linear(c_bottom, z_dim)
        self.from_z = nn.Linear(z_dim, self.core.flat_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.core.encode_to_bottom(x)
        h_vec = self._pool(h).flatten(1)  # (B, c_bottom)
        z = self.to_z(h_vec)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"decode expects z as torch.Tensor, got {type(z)}")
        if z.ndim != 2:
            raise ValueError(f"decode expects z shape (B, z_dim), got {tuple(z.shape)}")
        if z.shape[1] != self.z_dim:
            raise ValueError(f"decode expects z_dim={self.z_dim}, got {z.shape[1]}")

        B = z.shape[0]
        c_bottom, Hb, Wb = self.core.bottom_shape
        h = self.from_z(z).view(B, c_bottom, Hb, Wb)
        x_logits = self.core.decode_from_bottom(h)
        return self._apply_out_activation(x_logits)


# ----------------------------
# Utilities
# ----------------------------

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)