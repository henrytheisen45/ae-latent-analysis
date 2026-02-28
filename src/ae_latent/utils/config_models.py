from __future__ import annotations

from typing import Any, Dict, Tuple


def _get_str(d: Dict[str, Any], key: str) -> str:
    if key not in d or not isinstance(d[key], str):
        raise ValueError(f"model.{key} must be a string")
    return d[key]


def _get_int(d: Dict[str, Any], key: str) -> int:
    if key not in d:
        raise ValueError(f"Missing required field: model.{key}")
    return int(d[key])


def _parse_img_shape(v: Any) -> Tuple[int, int]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError("model.img_shape must be [H, W]")
    h, w = int(v[0]), int(v[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid model.img_shape: {v}")
    return h, w


def resolve_model_vector_latent_ae(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate + normalize the `model:` block for VectorLatentAE.

    Returns a NEW dict (does not mutate input).
    """
    name = _get_str(model, "name")
    if name != "VectorLatentAE":
        raise ValueError(f"Expected model.name='VectorLatentAE', got {name!r}")

    h, w = _parse_img_shape(model.get("img_shape"))

    out: Dict[str, Any] = dict(model)  # shallow copy
    out["name"] = name
    out["in_channels"] = _get_int(model, "in_channels")
    out["img_shape"] = [h, w]
    out["z_dim"] = _get_int(model, "z_dim")
    out["base_channels"] = _get_int(model, "base_channels")
    out["num_levels"] = _get_int(model, "num_levels")
    out["gn_max_groups"] = _get_int(model, "gn_max_groups")

    # Optional invariant (if you truly require this; otherwise delete)
    nl = int(out["num_levels"])
    if (h % (2**nl) != 0) or (w % (2**nl) != 0):
        raise ValueError(
            f"model.img_shape {h}x{w} must be divisible by 2**num_levels={2**nl} "
            f"(num_levels={nl})."
        )

    return out


def resolve_model_block(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch based on model.name.
    """
    if "name" not in model:
        raise ValueError("model.name is required")

    name = str(model["name"])
    if name == "VectorLatentAE":
        return resolve_model_vector_latent_ae(model)

    raise ValueError(f"Unknown model.name={name!r}. Add a resolver in config_models.py.")