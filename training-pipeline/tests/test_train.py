import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train


class RandomDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        noise = random.random() + float(np.random.random()) + torch.rand(1).item()
        return torch.tensor([index / 8, noise, 1.0]), index % 2


@pytest.fixture
def small_training(monkeypatch):
    def loaders(data_dir, batch_size, num_workers, download=True):
        assert download is False
        return (
            DataLoader(RandomDataset(), batch_size=batch_size, shuffle=True),
            DataLoader(RandomDataset(), batch_size=batch_size),
        )
    monkeypatch.setattr(train, "create_dataloaders", loaders)
    monkeypatch.setattr(train, "create_model", lambda spec: nn.Linear(3, 2))


def train_args(output, epochs, *extra, config_overrides=None):
    config = output.parent / f"{output.name}.json"
    values = {
        "epochs": epochs,
        "batch_size": 4,
        "device": "cpu",
        "download": False,
        "seed": 42,
        "model": {
            "target": "torch.nn:Linear",
            "params": {"in_features": 3, "out_features": 2},
        },
        "optimizer": {
            "target": "torch.optim:Adam",
            "params": {"lr": 0.001},
        },
        "scheduler": {
            "target": "torch.optim.lr_scheduler:CosineAnnealingLR",
            "params": {"T_max": 2, "eta_min": 0.0001},
        },
    }
    values.update(config_overrides or {})
    config.write_text(json.dumps(values))
    return train.parse_args([
        "--config", str(config), "--output-dir", str(output), *extra,
    ])


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_resume_matches_continuous_training_with_scheduler_and_rng(tmp_path, small_training):
    metadata = {"source": "local-test"}
    train.run_training(train_args(tmp_path / "full", 2), metadata=metadata)
    config = json.loads((tmp_path / "full/config.json").read_text())
    assert config["metadata"] == metadata == {"source": "local-test"}
    assert config["device"] == "cpu" and config["gpu"] is None
    train.run_training(train_args(tmp_path / "first", 1))
    train.run_training(train_args(
        tmp_path / "resumed", 2, "--resume", str(tmp_path / "first/last.pt"),
    ))
    full = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert full["epoch"] == resumed["epoch"] == 2
    assert_tree_equal(full["model_state_dict"], resumed["model_state_dict"])
    assert_tree_equal(full["optimizer_state_dict"], resumed["optimizer_state_dict"])
    assert_tree_equal(full["scheduler_state_dict"], resumed["scheduler_state_dict"])
    assert full["best_accuracy"] == resumed["best_accuracy"]
    metrics = [json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in metrics] == [2]
    assert metrics[0]["learning_rates"] == [0.00055]
    assert json.loads((tmp_path / "resumed/config.json").read_text())["seed"] == 42


def test_checkpoint_interval_keeps_all_metrics_and_final_checkpoint(tmp_path, small_training):
    class RecordingPublisher:
        def __init__(self):
            self.config = None
            self.checkpoints = []
            self.metrics_only = []

        def publish_config(self, config):
            self.config = config

        def publish_epoch(self, snapshot, metrics, is_best):
            payload = snapshot()
            assert payload["epoch"] == metrics["epoch"]
            self.checkpoints.append(metrics["epoch"])

        def publish_metrics(self, metrics):
            self.metrics_only.append(metrics["epoch"])

        def check(self):
            pass

    publisher = RecordingPublisher()
    train.run_training(
        train_args(tmp_path / "interval", 5, config_overrides={"checkpoint_interval": 2}),
        publisher=publisher,
    )
    assert publisher.config["checkpoint_interval"] == 2
    assert publisher.checkpoints == [2, 4, 5]
    assert publisher.metrics_only == [1, 3]


def test_resume_rejects_conflicting_batch_size(tmp_path, small_training):
    train.run_training(train_args(tmp_path / "first", 1))
    with pytest.raises(ValueError, match="batch_size"):
        train.run_training(train_args(
            tmp_path / "second", 2, "--resume", str(tmp_path / "first/last.pt"),
            config_overrides={"batch_size": 2},
        ))


def test_resume_rejects_already_completed_target(tmp_path, small_training):
    train.run_training(train_args(tmp_path / "first", 1))
    with pytest.raises(ValueError, match="epochs"):
        train.run_training(train_args(
            tmp_path / "second", 1, "--resume", str(tmp_path / "first/last.pt"),
        ))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("optimizer", {"target": "torch.optim:SGD", "params": {"lr": 0.001}}),
        ("scheduler", None),
    ],
)
def test_resume_rejects_conflicting_training_recipe(
    tmp_path, small_training, field, replacement,
):
    train.run_training(train_args(tmp_path / "first", 1))
    args = train_args(
        tmp_path / "second", 2,
        "--resume", str(tmp_path / "first/last.pt"),
    )
    setattr(args, field, replacement)
    with pytest.raises(ValueError, match=field):
        train.run_training(args)


def base_config(**overrides):
    config = {
        "model": {"target": "models.simple_cnn:SimpleCNN", "params": {}},
        "optimizer": {"target": "torch.optim:Adam", "params": {"lr": 0.001}},
        "scheduler": None,
    }
    config.update(overrides)
    return config


def test_config_cli_overrides_and_unknown_keys(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(base_config(epochs=3, download=False, batch_size=8)))
    args = train.parse_args(["--config", str(config), "--epochs", "5"])
    assert (args.epochs, args.batch_size, args.download) == (5, 8, False)
    config.write_text(json.dumps(base_config(epohcs=3)))
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(config)])


@pytest.mark.parametrize("option", ["--batch-size", "--checkpoint-interval", "--seed"])
def test_training_recipe_cli_overrides_are_rejected(tmp_path, option):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(base_config()))
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(config), option, "2"])


@pytest.mark.parametrize(
    "config",
    [{"num_workers": -1}, {"checkpoint_interval": 0}, {"epochs": "3"}, {"download": "false"}],
)
def test_config_rejects_invalid_types_and_values(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base_config(**config)))
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(path)])
