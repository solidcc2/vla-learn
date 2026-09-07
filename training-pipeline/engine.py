"""Reusable training and evaluation loops."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

Batch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float
    samples: int


def _run_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Metrics:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
       
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("the data loader did not produce any samples")
    return Metrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
    )


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Metrics:
    return _run_epoch(model, batches, criterion, device, optimizer)


def evaluate(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
) -> Metrics:
    return _run_epoch(model, batches, criterion, device, optimizer=None)
