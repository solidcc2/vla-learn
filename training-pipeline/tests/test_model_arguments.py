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
        "optimizer": {"target": "torch.optim:Adam", "params": {"lr": 0.001}},
        "scheduler": None,
    }
    config.update(overrides)
    return config


def test_training_requires_config() -> None:
    with pytest.raises(SystemExit):
        train.parse_args([])


def test_training_reads_explicit_model_target(tmp_path) -> None:
    config = tmp_path / "train.json"
    config.write_text(json.dumps(training_config(
        model={"target": "models.simple_cnn_soft_medoid:SimpleCNNWithSoftMedoid",
               "params": {"num_classes": 3}},
    )))
    args = train.parse_args(["--config", str(config)])
    assert args.model["target"].endswith(":SimpleCNNWithSoftMedoid")
    assert args.model["params"] == {"num_classes": 3}


def test_evaluation_uses_model_recorded_in_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    model_spec = {"target": "torch.nn:Linear", "params": {"in_features": 3, "out_features": 2}}
    payload = {"config": {"model": model_spec}}
    model = nn.Linear(3, 2)
    seen = {}
    monkeypatch.setattr(evaluate, "read_checkpoint", lambda path: payload)

    def build_model(spec):
        seen["spec"] = spec
        return model

    monkeypatch.setattr(evaluate, "create_model", build_model)
    monkeypatch.setattr(
        evaluate, "create_dataloaders",
        lambda *args, **kwargs: (None, [(torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))]),
    )
    monkeypatch.setattr(
        evaluate, "restore_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(epoch=4),
    )
    monkeypatch.setattr(
        evaluate, "evaluate",
        lambda *args, **kwargs: SimpleNamespace(loss=1.0, accuracy=0.5, samples=1),
    )
    state, metrics = evaluate.run_evaluation(evaluate.parse_args([str(checkpoint), "--device", "cpu"]))
    assert seen["spec"] == model_spec
    assert state.epoch == 4 and metrics.samples == 1
