"""CIFAR-10 training with offline data and epoch-boundary resume."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
from typing import Protocol, Callable

import numpy as np
import torch
from torch import nn

from checkpoint import (
    read_checkpoint, restore_checkpoint, save_checkpoint, snapshot_checkpoint,
    validate_checkpoint_config,
)
from components import create_model, create_optimizer, create_scheduler
from data import create_dataloaders
from engine import evaluate, train_one_epoch
from persistence.files import atomic_path, write_json
from training_config import load_training_config


class Publisher(Protocol):
    def publish_config(self, config: dict) -> None: ...
    def publish_epoch(self, snapshot: Callable[[], dict], metrics: dict, is_best: bool) -> None: ...
    def check(self) -> None: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured model on CIFAR-10")
    parser.add_argument("--config", type=Path, required=True,
                        help="Training JSON; explicit CLI options take precedence")
    parser.add_argument("--epochs", type=int, default=20, help="Target total epochs, including epochs before resume")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--data-version", default="cifar10", help="Dataset identity checked on resume")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, help="Local checkpoint path")
    known, _ = parser.parse_known_args(argv)
    try:
        parser.set_defaults(**load_training_config(known.config))
    except ValueError as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    for field in ("data_dir", "output_dir", "resume"):
        value = getattr(args, field)
        if isinstance(value, str):
            setattr(args, field, Path(value))
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
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
    resume_payload = read_checkpoint(args.resume) if args.resume else None
    compatibility = {
        "model": args.model, "optimizer": args.optimizer, "scheduler": args.scheduler,
        "data_version": args.data_version, "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if resume_payload is not None:
        validate_checkpoint_config(resume_payload, compatibility)
    effective_seed = resume_payload["config"]["seed"] if resume_payload is not None else args.seed
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    device = choose_device(args.device)
    train_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, download=args.download,
    )
    model = create_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args.optimizer, model.parameters())
    scheduler = create_scheduler(args.scheduler, optimizer)
    best_accuracy = -1.0
    start_epoch = 1
    if resume_payload is not None:
        state = restore_checkpoint(
            resume_payload, model, optimizer, scheduler, device, restore_rng=True,
        )
        start_epoch, best_accuracy = state.epoch + 1, state.best_accuracy
        if start_epoch > args.epochs:
            raise ValueError(f"--epochs must exceed the completed checkpoint epoch ({state.epoch})")
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        print(f"Resuming at epoch={start_epoch}; learning_rates={learning_rates}", flush=True)

    output = args.output_dir
    if publisher is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory must be empty; use a new run directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
    config = {
        **compatibility,
        "epochs": args.epochs, "seed": effective_seed,
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
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        metrics = {
            "epoch": epoch, "learning_rates": learning_rates,
            "train": asdict(train_metrics), "test": asdict(test_metrics),
        }
        is_best = test_metrics.accuracy > best_accuracy
        best_accuracy = max(best_accuracy, test_metrics.accuracy)
        if scheduler is not None:
            scheduler.step()
        if publisher:
            publisher.publish_epoch(
                lambda: snapshot_checkpoint(
                    model, optimizer, scheduler, epoch, best_accuracy, config=config,
                ),
                metrics, is_best,
            )
        else:
            save_checkpoint(
                output / "last.pt", model, optimizer, scheduler,
                epoch, best_accuracy, config=config,
            )
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
