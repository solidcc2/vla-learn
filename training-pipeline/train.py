"""CIFAR-10 training with offline data and epoch-boundary resume."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
from typing import Protocol, Callable

import numpy as np
import torch
from torch import nn

from checkpoint import load_checkpoint, save_checkpoint, snapshot_checkpoint
from data import create_dataloaders
from engine import evaluate, train_one_epoch
from model import SimpleCNN
from persistence.files import atomic_path, write_json


class Publisher(Protocol):
    def publish_config(self, config: dict) -> None: ...
    def publish_epoch(self, snapshot: Callable[[], dict], metrics: dict, is_best: bool) -> None: ...
    def check(self) -> None: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CNN on CIFAR-10")
    parser.add_argument("--config", type=Path, help="Training JSON; explicit CLI options take precedence")
    parser.add_argument("--epochs", type=int, default=20, help="Target total epochs, including epochs before resume")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--data-version", default="cifar10", help="Dataset identity checked on resume")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, help="Local checkpoint path; optimizer learning rate is preserved")
    known, _ = parser.parse_known_args(argv)
    if known.config:
        try:
            config = json.loads(known.config.read_text())
            if not isinstance(config, dict):
                raise ValueError("training config must be a JSON object")
            allowed = {
                "epochs": int, "batch_size": int, "learning_rate": (int, float),
                "data_dir": str, "data_version": str, "output_dir": str,
                "num_workers": int, "device": str, "seed": int,
                "download": bool, "resume": str,
            }
            for key, value in config.items():
                expected = allowed.get(key)
                if expected is None or type(value) not in (expected if isinstance(expected, tuple) else (expected,)):
                    raise ValueError(f"Unknown key or invalid type in training config: {key}")
            parser.set_defaults(**config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive and finite")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("num-workers must be nonnegative and seed must be in [0, 2**32)")
    if args.device not in ("auto", "cpu", "cuda"):
        parser.error("device must be auto, cpu or cuda")
    if not args.data_version:
        parser.error("data-version must not be empty")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def run_training(args: argparse.Namespace, publisher: Publisher | None = None, metadata: dict | None = None) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, download=args.download,
    )
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    compatibility = {
        "model": "SimpleCNN", "data_version": args.data_version,
        "batch_size": args.batch_size, "num_workers": args.num_workers, "seed": args.seed,
    }
    best_accuracy = -1.0
    start_epoch = 1
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, device,
                                restore_rng=True, expected_config=compatibility)
        start_epoch, best_accuracy = state.epoch + 1, state.best_accuracy
        if start_epoch > args.epochs:
            raise ValueError(f"--epochs must exceed the completed checkpoint epoch ({state.epoch})")
        print(f"Resuming at epoch={start_epoch}; optimizer learning_rate={optimizer.param_groups[0]['lr']}", flush=True)

    output = args.output_dir
    if publisher is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory must be empty; use a new run directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
    config = {
        **compatibility,
        "epochs": args.epochs, "learning_rate": optimizer.param_groups[0]["lr"],
        "device": str(device), "download": args.download,
        "data_dir": str(args.data_dir), "output_dir": str(output),
        "resume": str(args.resume) if args.resume else None,
        "torch_version": str(torch.__version__),
        "metadata": metadata if metadata is not None else {},
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    if publisher:
        publisher.publish_config(config)
    else:
        write_json(output / "config.json", config)

    for epoch in range(start_epoch, args.epochs + 1):
        if publisher:
            publisher.check()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        metrics = {"epoch": epoch, "train": asdict(train_metrics), "test": asdict(test_metrics)}
        is_best = test_metrics.accuracy > best_accuracy
        best_accuracy = max(best_accuracy, test_metrics.accuracy)
        if publisher:
            publisher.publish_epoch(
                lambda: snapshot_checkpoint(model, optimizer, epoch, best_accuracy, config=config),
                metrics, is_best,
            )
        else:
            save_checkpoint(output / "last.pt", model, optimizer, epoch, best_accuracy, config=config)
            if is_best:
                with atomic_path(output / "best.pt") as temporary:
                    shutil.copyfile(output / "last.pt", temporary)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics, allow_nan=False), flush=True)
    return metrics


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
