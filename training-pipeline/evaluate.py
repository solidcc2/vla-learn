"""Evaluate a saved CIFAR-10 checkpoint using its recorded model configuration."""

import argparse
from pathlib import Path

from torch import nn

from checkpoint import read_checkpoint, restore_checkpoint
from components import create_model
from data import create_dataloaders
from engine import evaluate
from train import choose_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run_evaluation(args: argparse.Namespace):
    payload = read_checkpoint(args.checkpoint)
    device = choose_device(args.device)
    _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, download=args.download
    )
    model = create_model(payload["config"]["model"]).to(device)
    state = restore_checkpoint(
        payload, model, optimizer=None, scheduler=None, device=device,
    )
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    return state, metrics


def main() -> None:
    state, metrics = run_evaluation(parse_args())
    print(
        f"checkpoint_epoch={state.epoch} loss={metrics.loss:.4f} "
        f"accuracy={metrics.accuracy:.2%} samples={metrics.samples}"
    )


if __name__ == "__main__":
    main()
