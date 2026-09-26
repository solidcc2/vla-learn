"""Versioned, atomic checkpoints for evaluation and epoch-boundary resume."""

from dataclasses import dataclass, field
import copy
from pathlib import Path
import random
import warnings

import numpy as np
import torch
from torch import nn

from persistence.files import atomic_path

FORMAT_VERSION = 4


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = type(value)((key, _cpu_copy(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_payload(model, optimizer, scheduler, epoch, best_metric, config):
    return {
        "format_version": FORMAT_VERSION, "epoch": epoch, "best_metric": best_metric,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config, "rng_state": capture_rng_state(),
    }


def snapshot_checkpoint(model, optimizer, scheduler, epoch, best_metric, *, config: dict):
    """Capture independent CPU storage before the next optimizer update."""
    return _cpu_copy(_checkpoint_payload(
        model, optimizer, scheduler, epoch, best_metric, config,
    ))


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    best_metric: dict
    config: dict = field(default_factory=dict)


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # Plain containers keep the checkpoint compatible with weights_only=True.
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state["cuda"]
    if torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    elif cuda_states or torch.cuda.is_available():
        warnings.warn("CUDA topology changed; exact random-state continuation is unavailable.", stacklevel=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: dict,
    *,
    config: dict,
) -> None:
    payload = _checkpoint_payload(model, optimizer, scheduler, epoch, best_metric, config)
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def read_checkpoint(path: Path) -> dict:
    """Read and validate a current-format checkpoint without restoring objects."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {version}; expected {FORMAT_VERSION}"
        )
    required = {
        "epoch", "best_metric", "model_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "config", "rng_state",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing field: {sorted(missing)[0]}")
    config = payload.get("config")
    config_fields = {"data", "model", "optimizer", "scheduler", "seed"}
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no complete training config")
    if missing_config := config_fields - config.keys():
        raise ValueError(f"Checkpoint config is missing field: {sorted(missing_config)[0]}")
    best_metric = payload["best_metric"]
    if (
        not isinstance(best_metric, dict)
        or set(best_metric) != {"name", "mode", "value", "epoch"}
        or best_metric["name"] != "validation.accuracy"
        or best_metric["mode"] != "max"
        or type(best_metric["value"]) not in (int, float)
        or not isinstance(best_metric["epoch"], int)
        or best_metric["epoch"] <= 0
    ):
        raise ValueError("Checkpoint has an invalid best_metric")

    scheduler_state = payload["scheduler_state_dict"]
    if (config["scheduler"] is None) != (scheduler_state is None):
        raise ValueError("Checkpoint scheduler config and state are inconsistent")
    return payload


def validate_checkpoint_config(payload: dict, expected_config: dict) -> None:
    """Reject resume settings that would change the continuation semantics."""
    saved_config = payload["config"]
    for key, expected in expected_config.items():
        if key not in saved_config:
            raise ValueError(f"Checkpoint config has no value for {key}")
        if saved_config[key] != expected:
            raise ValueError(
                f"Resume config mismatch for {key}: "
                f"saved={saved_config[key]!r}, requested={expected!r}"
            )


def restore_checkpoint(
    payload: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    *,
    restore_rng: bool = False,
) -> CheckpointState:
    config = payload["config"]
    model.to(device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return CheckpointState(
        int(payload["epoch"]), copy.deepcopy(payload["best_metric"]), config,
    )
