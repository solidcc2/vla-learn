import pytest
import torch
from torch import nn

from checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_training_state(tmp_path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "model.pt"
    expected_weight = model.weight.detach().clone()
    save_checkpoint(path, model, optimizer, epoch=3, best_accuracy=0.75)
    with torch.no_grad():
        model.weight.zero_()
    state = load_checkpoint(path, model, optimizer, torch.device("cpu"))
    assert state.epoch == 3
    assert state.best_accuracy == 0.75
    assert torch.equal(model.weight, expected_weight)


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, 1, 0.5)
    original = path.read_bytes()

    def broken_save(payload, target):
        with open(target, "wb") as stream:
            stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, model, optimizer, 2, 0.6)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_old_checkpoint_restores_with_warning(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "old.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 3, "best_accuracy": 0.75,
    }, path)
    with pytest.warns(UserWarning) as warnings:
        state = load_checkpoint(path, model, optimizer, torch.device("cpu"), restore_rng=True)
    assert any("random" in str(warning.message) for warning in warnings)
    assert state.epoch == 3


def test_unknown_checkpoint_version_is_rejected(tmp_path):
    model = nn.Linear(3, 2)
    path = tmp_path / "future.pt"
    torch.save({"format_version": 999}, path)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path, model, None, torch.device("cpu"))


def test_snapshot_owns_model_optimizer_and_rng_storage():
    from checkpoint import snapshot_checkpoint
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    snapshot = snapshot_checkpoint(model, optimizer, 1, 0.5)
    weight = snapshot["model_state_dict"]["weight"].clone()
    moment = snapshot["optimizer_state_dict"]["state"][0]["exp_avg"].clone()
    with torch.no_grad():
        model.weight.add_(20)
        optimizer.state[model.weight]["exp_avg"].add_(20)
    assert torch.equal(snapshot["model_state_dict"]["weight"], weight)
    assert torch.equal(snapshot["optimizer_state_dict"]["state"][0]["exp_avg"], moment)
