import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import evaluate
import train


def training_config(**overrides):
    config = {
        "model": {"target": "models.simple_cnn:SimpleCNN", "params": {}},
        "data": {"target": "data_modules.cifar:CIFARDataModule", "params": {}},
        "optimizer": {"target": "torch.optim:Adam", "params": {"lr": 0.001}},
        "scheduler": None,
    }
    config.update(overrides)
    return config


def test_training_requires_config() -> None:
    with pytest.raises(SystemExit):
        train.parse_args([])


def test_training_config_requires_data(tmp_path) -> None:
    config = training_config()
    del config["data"]
    path = tmp_path / "missing-data.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(path)])


def test_training_reads_explicit_model_target(tmp_path) -> None:
    config = tmp_path / "train.json"
    config.write_text(json.dumps(training_config(
        model={"target": "models.simple_cnn_soft_medoid:SimpleCNNWithSoftMedoid",
               "params": {"num_classes": 3}},
    )))
    args = train.parse_args(["--config", str(config)])
    assert isinstance(args, dict)
    assert args["model"] == {
        "target": "models.simple_cnn_soft_medoid:SimpleCNNWithSoftMedoid",
        "params": {"num_classes": 3},
    }


def test_evaluation_uses_model_and_test_data_recorded_in_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint_config = {
        "model": {
            "target": "torch.nn:Linear",
            "params": {"in_features": 3, "out_features": 2},
        },
        "data": {"target": "tests.fake:DataModule", "params": {}},
    }
    payload = SimpleNamespace(config=checkpoint_config)
    model = nn.Linear(3, 2)
    seen = {}
    monkeypatch.setattr(evaluate, "read_checkpoint", lambda path: payload)

    def build_model(config):
        seen["model_config"] = config
        return model

    monkeypatch.setattr(evaluate, "create_model", build_model)
    class TestDataModule:
        def create_test_loader(self, *args, **kwargs):
            seen["test_loader"] = True
            return [(torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))]

    def build_data_module(config):
        seen["data_config"] = config
        return TestDataModule()

    monkeypatch.setattr(evaluate, "create_data_module", build_data_module)
    monkeypatch.setattr(
        evaluate, "restore_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(epoch=4),
    )
    monkeypatch.setattr(
        evaluate, "evaluate",
        lambda *args, **kwargs: SimpleNamespace(loss=1.0, accuracy=0.5, samples=1),
    )
    state, metrics = evaluate.run_evaluation(evaluate.parse_args([str(checkpoint), "--device", "cpu"]))
    assert seen["model_config"] is checkpoint_config
    assert seen["data_config"] is checkpoint_config
    assert seen["test_loader"] is True
    assert state.epoch == 4 and metrics.samples == 1
