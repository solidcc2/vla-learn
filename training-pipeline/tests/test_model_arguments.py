import json
import pytest

import evaluate
import train


def test_training_model_defaults_to_baseline_and_can_select_soft_medoid() -> None:
    assert train.parse_args([]).model == "simple_cnn"
    assert train.parse_args(["--model", "simple_cnn_soft_medoid"]).model == "simple_cnn_soft_medoid"


def test_training_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        train.parse_args(["--model", "missing"])


def test_training_rejects_unknown_model_from_config(tmp_path) -> None:
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"model": "missing"}))
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(config)])


def test_evaluation_accepts_model_selection(tmp_path) -> None:
    args = evaluate.parse_args([str(tmp_path / "checkpoint.pt"), "--model", "simple_cnn_soft_medoid"])
    assert args.model == "simple_cnn_soft_medoid"
