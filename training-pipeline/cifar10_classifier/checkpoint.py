"""Checkpoint persistence for training and evaluation."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    best_accuracy: float


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> CheckpointState:
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return CheckpointState(
        epoch=int(payload["epoch"]),
        best_accuracy=float(payload["best_accuracy"]),
    )
