"""
Convolutional autoencoder with a vector bottleneck.

This file defines a single baseline architecture used for latent geometry
experiments. The goal is to vary z_dim while keeping the convolutional
encoder/decoder body fixed, so changes in intrinsic dimension estimates
can be attributed to the bottleneck rather than architectural drift.

Architecture
------------
Encoder:
    x ∈ R^{BxCxHxW}
        → h_bottom ∈ R^{BxCbxHbxWb}

Reduction:
    flatten(h_bottom) ∈ R^{Bx(Cb·Hb·Wb)}

Bottleneck:
    z = Linear(flat_dim → z_dim)

Decoder:
    Linear(z_dim → flat_dim)
        → reshape to (B, Cb, Hb, Wb)
        → convolutional decoder
        → x_logits ∈ R^{BxCxHxW}
        → optional output activation

Across z_dim sweeps
-------------------
When {in_channels, img_shape, base_channels, num_levels, gn_max_groups}
are fixed, the convolutional body is identical across runs. Only the
linear bottleneck maps (to_z, from_z) change. This isolates the effect
of bottleneck dimension.

Shape constraint
----------------
img_shape = (H, W) must be divisible by 2**num_levels in both dimensions.
Padding/cropping should be handled in the dataloader.

Normalization
-------------
GroupNorm + SiLU is used throughout. Batch statistics are avoided to
maintain stability at small batch sizes.

gn_max_groups
-------------
Maximum (preferred) number of GroupNorm groups. The actual number of
groups for a layer is the largest divisor of out_channels not exceeding
gn_max_groups (fallback to 1 if necessary).

Output activation
-----------------
tanh     : for inputs normalized to [-1, 1]
sigmoid  : for inputs in [0, 1]
none     : return raw logits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn

OutAct = Literal["tanh", "sigmoid", "none"]


# ----------------------------
# Validation and small helpers
# ----------------------------

def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _validate_nonneg_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")


def _validate_img_shape(img_shape: Tuple[int, int]) -> Tuple[int, int]:
    if not (isinstance(img_shape, tuple) and len(img_shape) == 2):
        raise TypeError(f"img_shape must be a tuple (H, W), got {img_shape!r}")
    H, W = img_shape
    if not (isinstance(H, int) and isinstance(W, int)):
        raise TypeError(f"img_shape must contain ints, got {img_shape!r}")
    if H <= 0 or W <= 0:
        raise ValueError(f"img_shape must be positive, got {img_shape!r}")
    return H, W


def _validate_out_activation(out_activation: OutAct) -> None:
    if out_activation not in ("tanh", "sigmoid", "none"):
        raise ValueError(f"out_activation must be one of ['tanh','sigmoid','none'], got {out_activation!r}")


def _pick_gn_groups(num_channels: int, gn_max_groups: int) -> int:
    """
    Pick the GroupNorm group count.

    Parameters
    ----------
    num_channels:
      Number of channels to be normalized (must be > 0).
    gn_max_groups:
      Maximum/preferred number of GN groups. The returned group count will be
      the largest divisor of num_channels that is <= gn_max_groups.
      Falls back to 1 (LayerNorm-ish behavior across channels) if needed.
    """
    _validate_positive_int("num_channels", num_channels)
    _validate_positive_int("gn_max_groups", gn_max_groups)

    g = min(gn_max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g


@dataclass(frozen=True)
class _ShapeSpec:
    H: int
    W: int
    div: int
    Hb: int
    Wb: int
    c_bottom: int
    flat_dim: int


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


# ----------------------------
# Building blocks
# ----------------------------

class ConvBlock(nn.Module):
    """Conv -> GN -> SiLU -> Conv -> GN -> SiLU"""
    def __init__(self, in_ch: int, out_ch: int, *, gn_max_groups: int = 8):
        super().__init__()
        _validate_positive_int("in_ch", in_ch)
        _validate_positive_int("out_ch", out_ch)
        g = _pick_gn_groups(out_ch, gn_max_groups)

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
    def __init__(self, in_ch: int, out_ch: int, *, gn_max_groups: int = 8):
        super().__init__()
        _validate_positive_int("in_ch", in_ch)
        _validate_positive_int("out_ch", out_ch)
        g = _pick_gn_groups(out_ch, gn_max_groups)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Upsample(nn.Module):
    """Upsample by 2 via transposed conv (kernel=4,stride=2,pad=1)."""
    def __init__(self, in_ch: int, out_ch: int, *, gn_max_groups: int = 8):
        super().__init__()
        _validate_positive_int("in_ch", in_ch)
        _validate_positive_int("out_ch", out_ch)
        g = _pick_gn_groups(out_ch, gn_max_groups)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Convolutional body: encoder + decoder
# ----------------------------

class _EncoderDecoderBody(nn.Module):
    """
    Convolutional encoder/decoder body (no bottleneck logic).

    Shapes:
      x:        (B, in_channels, H, W)
      h_bottom: (B, c_bottom, Hb, Wb)
      x_logits: (B, in_channels, H, W)   (pre output activation)
    """
    def __init__(
        self,
        *,
        in_channels: int,
        img_shape: Tuple[int, int],
        base_channels: int,
        num_levels: int,
        gn_max_groups: int,
    ):
        super().__init__()
        _validate_positive_int("in_channels", in_channels)
        _validate_positive_int("gn_max_groups", gn_max_groups)

        spec = _validate_and_compute_shapes(img_shape, base_channels, num_levels)

        self.in_channels = in_channels
        self.img_shape = (spec.H, spec.W)
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.gn_max_groups = gn_max_groups

        # Exposed helpers for debugging/analysis
        self.bottom_shape: Tuple[int, int, int] = (spec.c_bottom, spec.Hb, spec.Wb)
        self.flat_dim: int = spec.flat_dim

        # Encoder
        ch0 = base_channels
        enc = [ConvBlock(in_channels, ch0, gn_max_groups=gn_max_groups)]
        c_in = ch0
        for level in range(num_levels):
            c_out = base_channels * (2 ** (level + 1))
            enc.append(Downsample(c_in, c_out, gn_max_groups=gn_max_groups))
            enc.append(ConvBlock(c_out, c_out, gn_max_groups=gn_max_groups))
            c_in = c_out
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = [ConvBlock(spec.c_bottom, spec.c_bottom, gn_max_groups=gn_max_groups)]
        c_in = spec.c_bottom
        for level in range(num_levels - 1, -1, -1):
            c_out = base_channels * (2 ** level)
            dec.append(Upsample(c_in, c_out, gn_max_groups=gn_max_groups))
            dec.append(ConvBlock(c_out, c_out, gn_max_groups=gn_max_groups))
            c_in = c_out
        self.decoder = nn.Sequential(*dec)

        self.out_conv = nn.Conv2d(ch0, in_channels, kernel_size=1, bias=True)

    # ---- checks ----
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

    # ---- transforms ----
    def encode_to_bottom(self, x: torch.Tensor) -> torch.Tensor:
        self.check_input(x)
        return self.encoder(x)

    def decode_from_bottom(self, h: torch.Tensor) -> torch.Tensor:
        self.check_bottom(h)
        x = self.decoder(h)
        return self.out_conv(x)  # logits


# ----------------------------
# Model: VectorLatentAE
# ----------------------------

class VectorLatentAE(nn.Module):
    """
    Vector bottleneck AE with flatten reduction.

    Parameters
    ----------
    in_channels:
      Input channels (e.g., 3 for RGB).
    img_shape:
      Input spatial size (H, W). Must satisfy divisibility by 2**num_levels.
    z_dim:
      Latent vector dimension.
    base_channels:
      Channels after the first conv block. Doubles each downsample level.
    num_levels:
      Number of stride-2 downsamples (and symmetric upsamples).
    out_activation:
      "tanh" | "sigmoid" | "none"
    gn_max_groups:
      Maximum/preferred GN group count. Actual groups per layer are chosen as the
      largest divisor of out_ch <= gn_max_groups.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        img_shape: Tuple[int, int],
        z_dim: int,
        base_channels: int = 32,
        num_levels: int = 4,
        out_activation: OutAct = "tanh",
        gn_max_groups: int = 8,
    ):
        _validate_positive_int("in_channels", in_channels)
        _validate_img_shape(img_shape)
        _validate_positive_int("z_dim", z_dim)
        _validate_positive_int("base_channels", base_channels)
        _validate_nonneg_int("num_levels", num_levels)
        _validate_out_activation(out_activation)
        _validate_positive_int("gn_max_groups", gn_max_groups)

        super().__init__()

        self.in_channels = in_channels
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.out_activation: OutAct = out_activation

        self.body = _EncoderDecoderBody(
            in_channels=in_channels,
            img_shape=img_shape,
            base_channels=base_channels,
            num_levels=num_levels,
            gn_max_groups=gn_max_groups,
        )

        # Useful for debugging/analysis
        self.bottom_shape = self.body.bottom_shape
        self.flat_dim = self.body.flat_dim

        # Bottleneck heads
        self.to_z = nn.Linear(self.flat_dim, z_dim)
        self.from_z = nn.Linear(z_dim, self.flat_dim)

    def _apply_out_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_activation == "tanh":
            return torch.tanh(x)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body.encode_to_bottom(x)
        return self.to_z(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"decode expects z as torch.Tensor, got {type(z)}")
        if z.ndim != 2:
            raise ValueError(f"decode expects z shape (B, z_dim), got {tuple(z.shape)}")
        if z.shape[1] != self.z_dim:
            raise ValueError(f"decode expects z_dim={self.z_dim}, got {z.shape[1]}")

        # Fail loudly: do not silently move tensors across devices.
        param_device = next(self.parameters()).device
        if z.device != param_device:
            raise RuntimeError(f"z is on {z.device}, but model parameters are on {param_device}. Move z explicitly.")

        B = z.shape[0]
        c_bottom, Hb, Wb = self.bottom_shape
        h = self.from_z(z).reshape(B, c_bottom, Hb, Wb)
        x_logits = self.body.decode_from_bottom(h)
        return self._apply_out_activation(x_logits)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        recon = self.decode(z)
        return (recon, z) if return_latent else recon

    @torch.no_grad()
    def get_latent_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience for analysis: returns detached z (B, z_dim).
        Preserves module's training/eval mode.
        """
        was_training = self.training
        try:
            self.eval()
            return self.encode(x).detach()
        finally:
            self.train(was_training)


# ----------------------------
# Utils
# ----------------------------

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)