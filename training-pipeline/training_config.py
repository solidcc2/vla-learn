"""Load and validate JSON training configuration."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "epochs": int,
    "batch_size": int,
    "checkpoint_interval": int,
    "data_dir": str,
    "data_version": str,
    "output_dir": str,
    "num_workers": int,
    "device": str,
    "seed": int,
    "download": bool,
    "resume": (str, type(None)),
    "model": dict,
    "optimizer": dict,
    "scheduler": (dict, type(None)),
}


def normalize_object_spec(value: Any, name: str, *, optional: bool = False) -> dict | None:
    """Return a canonical target/params object specification."""
    if value is None and optional:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - {"target", "params"}
    if unknown:
        raise ValueError(f"Unknown key in {name}: {sorted(unknown)[0]}")
    target = value.get("target")
    params = value.get("params", {})
    if not isinstance(target, str) or not target:
        raise ValueError(f"{name}.target must be a non-empty string")
    if not isinstance(params, dict):
        raise ValueError(f"{name}.params must be an object")
    return {"target": target, "params": deepcopy(params)}


def load_training_config(path: Path) -> dict:
    """Read a training JSON and return a validated, normalized dictionary."""
    try:
        config = json.loads(path.read_text())
    except OSError as error:
        raise ValueError(str(error)) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid training JSON: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("training config must be a JSON object")
    for required in ("model", "optimizer"):
        if required not in config:
            raise ValueError(f"training config must define {required}")
    for key, value in config.items():
        expected = _FIELD_TYPES.get(key)
        if expected is None:
            raise ValueError(f"Unknown key in training config: {key}")
        expected_types = expected if isinstance(expected, tuple) else (expected,)
        if type(value) not in expected_types:
            raise ValueError(f"Invalid type for training config key: {key}")
    normalized = deepcopy(config)
    normalized["model"] = normalize_object_spec(config["model"], "model")
    normalized["optimizer"] = normalize_object_spec(config["optimizer"], "optimizer")
    normalized["scheduler"] = normalize_object_spec(
        config.get("scheduler"), "scheduler", optional=True,
    )
    return normalized
