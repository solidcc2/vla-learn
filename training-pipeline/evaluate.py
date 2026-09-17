"""Evaluate a saved CIFAR-10 CNN checkpoint."""

import argparse
from pathlib import Path

import torch
from torch import nn

from checkpoint import load_checkpoint
from data import create_dataloaders
from engine import evaluate
from models import MODEL_NAMES, create_model
from train import choose_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", choices=MODEL_NAMES, default="simple_cnn")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    device = choose_device(args.device)
    _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, download=args.download
    )
    model = create_model(args.model).to(device)
    state = load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(
        f"checkpoint_epoch={state.epoch} loss={metrics.loss:.4f} "
        f"accuracy={metrics.accuracy:.2%} samples={metrics.samples}"
    )


if __name__ == "__main__":
    main()
