"""Command-line entry point for CIFAR-10 training."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from cifar10_classifier.checkpoint import save_checkpoint
from cifar10_classifier.data import create_dataloaders
from cifar10_classifier.engine import evaluate, train_one_epoch
from cifar10_classifier.model import SimpleCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.2%} "
            f"test_loss={test_metrics.loss:.4f} "
            f"test_acc={test_metrics.accuracy:.2%}"
        )
        current_best = max(best_accuracy, test_metrics.accuracy)
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, epoch, current_best
        )
        if test_metrics.accuracy > best_accuracy:
            best_accuracy = test_metrics.accuracy
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_accuracy,
            )


if __name__ == "__main__":
    main()
