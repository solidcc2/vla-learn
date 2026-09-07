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
    monkeypatch.setattr(train, "SimpleCNN", lambda: nn.Linear(3, 2))


def train_args(output, epochs, *extra):
    return train.parse_args([
        "--output-dir", str(output), "--epochs", str(epochs),
        "--device", "cpu", "--batch-size", "4", "--no-download", *extra,
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


def test_resume_matches_continuous_training_with_adam_and_rng(tmp_path, small_training):
    metadata = {"source": "local-test"}
    train.run_training(train_args(tmp_path / "full", 2), metadata=metadata)
    config = json.loads((tmp_path / "full/config.json").read_text())
    assert config["metadata"] == metadata == {"source": "local-test"}
    assert config["device"] == "cpu" and config["gpu"] is None
    train.run_training(train_args(tmp_path / "first", 1))
    train.run_training(train_args(
        tmp_path / "resumed", 2, "--resume", str(tmp_path / "first/last.pt"),
        "--learning-rate", "0.9",
    ))
    full = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert full["epoch"] == resumed["epoch"] == 2
    assert_tree_equal(full["model_state_dict"], resumed["model_state_dict"])
    assert_tree_equal(full["optimizer_state_dict"], resumed["optimizer_state_dict"])
    assert full["best_accuracy"] == resumed["best_accuracy"]
    metrics = [json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in metrics] == [2]
    assert json.loads((tmp_path / "resumed/config.json").read_text())["learning_rate"] == 0.001


def test_resume_rejects_conflicting_batch_size(tmp_path, small_training):
    train.run_training(train_args(tmp_path / "first", 1))
    with pytest.raises(ValueError, match="batch_size"):
        train.run_training(train_args(
            tmp_path / "second", 2, "--resume", str(tmp_path / "first/last.pt"),
            "--batch-size", "2",
        ))


def test_resume_rejects_already_completed_target(tmp_path, small_training):
    train.run_training(train_args(tmp_path / "first", 1))
    with pytest.raises(ValueError, match="epochs"):
        train.run_training(train_args(
            tmp_path / "second", 1, "--resume", str(tmp_path / "first/last.pt"),
        ))


def test_config_cli_overrides_and_unknown_keys(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"epochs": 3, "download": False, "batch_size": 8}))
    args = train.parse_args(["--config", str(config), "--epochs", "5"])
    assert (args.epochs, args.batch_size, args.download) == (5, 8, False)
    config.write_text('{"epohcs": 3}')
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(config)])


@pytest.mark.parametrize("config", [{"num_workers": -1}, {"epochs": "3"}, {"download": "false"}])
def test_config_rejects_invalid_types_and_values(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(path)])
