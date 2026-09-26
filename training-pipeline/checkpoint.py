"""Versioned, atomic checkpoints for evaluation and epoch-boundary resume."""

from dataclasses import dataclass
import copy
from pathlib import Path
import random
import warnings

import numpy as np
import torch
from torch import nn

from engine import BestMetric
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


@dataclass(frozen=True)
class CheckpointPayload:
    format_version: int
    epoch: int
    best_metric: BestMetric
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict | None
    config: dict
    rng_state: dict

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "epoch": self.epoch,
            "best_metric": self.best_metric.to_dict(),
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "config": copy.deepcopy(self.config),
            "rng_state": self.rng_state,
        }


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: BestMetric,
    config: dict,
) -> CheckpointPayload:
    return CheckpointPayload(
        format_version=FORMAT_VERSION,
        epoch=epoch,
        best_metric=best_metric,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        config=config,
        rng_state=capture_rng_state(),
    )


def snapshot_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: BestMetric,
    *,
    config: dict,
) -> dict:
    """Capture independent CPU storage before the next optimizer update."""
    payload = _checkpoint_payload(
        model, optimizer, scheduler, epoch, best_metric, config,
    )
    return _cpu_copy(payload.to_dict())


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    best_metric: BestMetric
    config: dict


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
    best_metric: BestMetric,
    *,
    config: dict,
) -> None:
    payload = _checkpoint_payload(
        model, optimizer, scheduler, epoch, best_metric, config,
    )
    with atomic_path(path) as temporary:
        torch.save(payload.to_dict(), temporary)


def read_checkpoint(path: Path) -> CheckpointPayload:
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

    config = payload["config"]
    config_fields = {"data", "model", "optimizer", "scheduler", "seed"}
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no complete training config")
    if missing_config := config_fields - config.keys():
        raise ValueError(f"Checkpoint config is missing field: {sorted(missing_config)[0]}")
    best_metric = BestMetric.from_dict(payload["best_metric"])
    scheduler_state = payload["scheduler_state_dict"]
    if (config["scheduler"] is None) != (scheduler_state is None):
        raise ValueError("Checkpoint scheduler config and state are inconsistent")
    return CheckpointPayload(
        format_version=version,
        epoch=int(payload["epoch"]),
        best_metric=best_metric,
        model_state_dict=payload["model_state_dict"],
        optimizer_state_dict=payload["optimizer_state_dict"],
        scheduler_state_dict=scheduler_state,
        config=copy.deepcopy(config),
        rng_state=payload["rng_state"],
    )


def validate_checkpoint_config(
    payload: CheckpointPayload,
    expected: dict,
) -> None:
    """Reject resume settings that would change the continuation semantics."""
    comparisons = (
        "data", "model", "optimizer", "scheduler",
        "data_version", "batch_size", "num_workers",
    )
    for name in comparisons:
        if name not in payload.config:
            raise ValueError(f"Checkpoint config has no value for {name}")
        if payload.config[name] != expected[name]:
            raise ValueError(
                f"Resume config mismatch for {name}: "
                f"saved={payload.config[name]!r}, requested={expected[name]!r}"
            )


def restore_checkpoint(
    payload: CheckpointPayload,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    *,
    restore_rng: bool = False,
) -> CheckpointState:
    model.to(device)
    model.load_state_dict(payload.model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(payload.optimizer_state_dict)
    if scheduler is not None:
        scheduler.load_state_dict(payload.scheduler_state_dict)
    if restore_rng:
        restore_rng_state(payload.rng_state)
    return CheckpointState(
        payload.epoch, payload.best_metric, copy.deepcopy(payload.config),
    )
