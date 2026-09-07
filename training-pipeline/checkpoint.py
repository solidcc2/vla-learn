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

FORMAT_VERSION = 2


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


def _checkpoint_payload(model, optimizer, epoch, best_accuracy, config):
    return {
        "format_version": FORMAT_VERSION, "epoch": epoch, "best_accuracy": best_accuracy,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "config": config or {}, "rng_state": capture_rng_state(),
    }


def snapshot_checkpoint(model, optimizer, epoch, best_accuracy, *, config=None):
    """Capture independent CPU storage before the next optimizer update."""
    return _cpu_copy(_checkpoint_payload(model, optimizer, epoch, best_accuracy, config))


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    best_accuracy: float
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
    epoch: int,
    best_accuracy: float,
    *,
    config: dict | None = None,
) -> None:
    payload = _checkpoint_payload(model, optimizer, epoch, best_accuracy, config)
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    restore_rng: bool = False,
    expected_config: dict | None = None,
) -> CheckpointState:
    # Keep RNG byte tensors on CPU. Module/optimizer loading moves their own state
    # to the device of the already constructed model, including Adam's step state.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    version = payload.get("format_version", 1)
    if version not in (1, FORMAT_VERSION):
        raise ValueError(f"Unsupported checkpoint format version: {version}")
    config = payload.get("config", {})
    for key, expected in (expected_config or {}).items():
        if key in config and config[key] != expected:
            raise ValueError(f"Resume config mismatch for {key}: saved={config[key]!r}, requested={expected!r}")
    model.to(device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        else:
            warnings.warn("Legacy checkpoint has no random state; exact continuation is unavailable.", stacklevel=2)
        if not config:
            warnings.warn("Checkpoint has no training config; resume compatibility cannot be checked.", stacklevel=2)
    return CheckpointState(int(payload["epoch"]), float(payload["best_accuracy"]), config)
