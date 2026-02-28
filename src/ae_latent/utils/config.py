from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Literal
import json
import time
import copy

from ae_latent.utils.config_models import resolve_model_block

try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load/save YAML configs. Install with: pip install pyyaml"
    ) from e


NormalizeMode = Literal["-1_1", "0_1"]


# ----------------------------
# YAML IO
# ----------------------------

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping/dict. Got: {type(data)}")
    return data


def dump_yaml(data: Dict[str, Any]) -> str:
    # Keep this stable and human-readable
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


# ----------------------------
# Parsing helpers
# ----------------------------

def _require_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in cfg or not isinstance(cfg[key], dict):
        raise ValueError(f"cfg must contain dict key: {key}")
    return cfg[key]


def _get_str(d: Dict[str, Any], key: str, default: Optional[str] = None) -> str:
    if key not in d:
        if default is None:
            raise ValueError(f"Missing required field: {key}")
        return str(default)
    v = d[key]
    if not isinstance(v, str):
        raise ValueError(f"{key} must be a string, got {type(v)}")
    return v


def _get_int(d: Dict[str, Any], key: str, default: Optional[int] = None) -> int:
    if key not in d:
        if default is None:
            raise ValueError(f"Missing required field: {key}")
        return int(default)
    return int(d[key])


def _get_float(d: Dict[str, Any], key: str, default: Optional[float] = None) -> float:
    if key not in d:
        if default is None:
            raise ValueError(f"Missing required field: {key}")
        return float(default)
    return float(d[key])


def _get_bool(d: Dict[str, Any], key: str, default: Optional[bool] = None) -> bool:
    if key not in d:
        if default is None:
            raise ValueError(f"Missing required field: {key}")
        return bool(default)
    return bool(d[key])


def _parse_normalize_mode(v: Any) -> NormalizeMode:
    if v is None:
        return "-1_1"
    if not isinstance(v, str):
        raise ValueError(f"data.normalize must be a string, got {type(v)}")
    s = v.strip()
    if s not in ("-1_1", "0_1"):
        raise ValueError(f"data.normalize must be '-1_1' or '0_1', got {v!r}")
    return s  # type: ignore[return-value]


# ----------------------------
# Run name derivation
# ----------------------------

def _sanitize(s: str) -> str:
    # Keep it filesystem friendly
    s = s.strip()
    s = s.replace(" ", "_")
    # remove path separators just in case
    s = s.replace("/", "_").replace("\\", "_")
    return s


def derive_run_stem(resolved: Dict[str, Any]) -> str:
    run = _require_dict(resolved, "run")
    data = _require_dict(resolved, "data")
    model = _require_dict(resolved, "model")

    base = _sanitize(_get_str(run, "run_name_base"))
    dataset = _sanitize(_get_str(data, "dataset").lower())
    z_dim = _get_int(model, "z_dim")

    return f"{base}_{dataset}_z{z_dim}"


def create_unique_run_dir(
    *,
    save_dir: Union[str, Path],
    run_stem: str,
    max_tries: int = 10_000,
) -> Path:
    """
    Create {save_dir}/{run_stem}--N atomically (exist_ok=False), picking the smallest
    available N. Safe enough for concurrent launches (retries on collision).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for n in range(max_tries):
        run_name = f"{run_stem}--{n}"
        run_dir = save_dir / run_name
        try:
            run_dir.mkdir(parents=True, exist_ok=False) 
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError(f"Could not allocate a unique run dir after {max_tries} tries for stem={run_stem!r}")


# ----------------------------
# Resolver
# ----------------------------

def resolve_config(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve a human-authored YAML spec into a fully-typed, validated config.

    Adds/normalizes:
      - data.val_split_seed default = run.seed
      - data.normalize validated
      - model.img_shape normalized to [H, W] ints
      - run.run_name_base required
      - (does NOT create run_dir; use prepare_run() for that)
    """
    cfg = copy.deepcopy(spec)

    run = _require_dict(cfg, "run")
    data = _require_dict(cfg, "data")
    model = _require_dict(cfg, "model")
    _require_dict(cfg, "train")
    _require_dict(cfg, "loss")

    # Required run fields
    run["seed"] = _get_int(run, "seed")
    run["save_dir"] = _get_str(run, "save_dir")
    run["run_name_base"] = _get_str(run, "run_name_base")

    # Model
    cfg["model"] = resolve_model_block(model)
   

    # Data basics
    data["dataset"] = _get_str(data, "dataset").lower()
    data["root"] = _get_str(data, "root")
    data["download"] = _get_bool(data, "download", True)
    data["batch_size"] = _get_int(data, "batch_size")
    data["num_workers"] = _get_int(data, "num_workers", 0)
    data["pin_memory"] = _get_bool(data, "pin_memory", True)
    data["persistent_workers"] = _get_bool(data, "persistent_workers", True)
    data["shuffle_train"] = _get_bool(data, "shuffle_train", True)
    data["drop_last"] = _get_bool(data, "drop_last", False)
    data["val_split"] = _get_float(data, "val_split", 0.1)

    # split seed separate from run.seed
    data["val_split_seed"] = _get_int(data, "val_split_seed", run["seed"])

    # normalization mode
    data["normalize"] = _parse_normalize_mode(data.get("normalize", "-1_1"))

    # Loss
    loss = _require_dict(cfg, "loss")
    loss_name = _get_str(loss, "name").lower().strip()
    if loss_name not in ("mse", "l1"):
        raise ValueError(f"loss.name must be 'mse' or 'l1', got {loss_name!r}")
    loss["name"] = loss_name

    # Train latent_reg shape validation
    train = _require_dict(cfg, "train")
    if "latent_reg" not in train or not isinstance(train["latent_reg"], dict):
        # Default it if missing
        train["latent_reg"] = {"kind": "none", "strength": 0.0, "params": {}}
    lr = train["latent_reg"]
    lr_kind = str(lr.get("kind", "none")).lower().strip()
    lr_strength = float(lr.get("strength", 0.0))
    lr_params = lr.get("params", {})
    if not isinstance(lr_params, dict):
        raise ValueError("train.latent_reg.params must be a dict (use {} for empty).")

    train["latent_reg"] = {
        "kind": lr_kind,
        "strength": lr_strength,
        "params": lr_params,
    }

    return cfg


def prepare_run(
    *,
    spec: Dict[str, Any],
    spec_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Resolve config and create a unique run directory.
    Writes:
      - spec.yaml  (the exact spec used)
      - resolved.json (fully resolved config + run.name/run.dir)
    Returns: (run_dir, resolved_cfg)
    """
    resolved = resolve_config(spec)

    run_stem = derive_run_stem(resolved)
    save_dir = Path(resolved["run"]["save_dir"])
    run_dir = create_unique_run_dir(save_dir=save_dir, run_stem=run_stem)

    # Attach derived run identity
    resolved["run"]["name"] = run_dir.name
    resolved["run"]["dir"] = str(run_dir.resolve())
    resolved["run"]["created_at"] = time.strftime("%Y-%m-%d_%H-%M-%S")
    if spec_path is not None:
        resolved["run"]["spec_path"] = str(Path(spec_path).resolve())

    # Write artifacts
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # spec.yaml (preserve user-authored spec)
    (run_dir / "spec.yaml").write_text(dump_yaml(spec), encoding="utf-8")

    # resolved.json
    (run_dir / "resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    return run_dir, resolved