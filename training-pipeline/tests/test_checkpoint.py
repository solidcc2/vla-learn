import pytest
import torch
from torch import nn

from checkpoint import (
    read_checkpoint, restore_checkpoint, save_checkpoint, snapshot_checkpoint,
)


def checkpoint_config(scheduler=None):
    return {
        "model": {"target": "torch.nn:Linear", "params": {"in_features": 3, "out_features": 2}},
        "optimizer": {"target": "torch.optim:SGD", "params": {"lr": 0.1}},
        "scheduler": scheduler,
        "seed": 42,
    }


def test_checkpoint_restores_model_optimizer_and_scheduler(tmp_path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler_spec = {
        "target": "torch.optim.lr_scheduler:StepLR",
        "params": {"step_size": 1, "gamma": 0.5},
    }
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    path = tmp_path / "model.pt"
    expected_weight = model.weight.detach().clone()
    save_checkpoint(
        path, model, optimizer, scheduler, epoch=3, best_accuracy=0.75,
        config=checkpoint_config(scheduler_spec),
    )
    with torch.no_grad():
        model.weight.zero_()
    optimizer.param_groups[0]["lr"] = 0.9
    restored_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    state = restore_checkpoint(
        read_checkpoint(path), model, optimizer, restored_scheduler, torch.device("cpu"),
    )
    assert state.epoch == 3
    assert state.best_accuracy == 0.75
    assert torch.equal(model.weight, expected_weight)
    assert optimizer.param_groups[0]["lr"] == 0.05
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "last.pt"
    config = checkpoint_config()
    save_checkpoint(path, model, optimizer, None, 1, 0.5, config=config)
    original = path.read_bytes()

    def broken_save(payload, target):
        with open(target, "wb") as stream:
            stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, model, optimizer, None, 2, 0.6, config=config)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("version", [1, 2, 999])
def test_noncurrent_checkpoint_version_is_rejected(tmp_path, version):
    path = tmp_path / "unsupported.pt"
    torch.save({"format_version": version}, path)
    with pytest.raises(ValueError, match="expected 3"):
        read_checkpoint(path)


def test_checkpoint_rejects_scheduler_config_state_mismatch(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "invalid.pt"
    save_checkpoint(path, model, optimizer, None, 1, 0.5, config=checkpoint_config())
    payload = torch.load(path, weights_only=True)
    payload["scheduler_state_dict"] = {"last_epoch": 1}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="inconsistent"):
        read_checkpoint(path)


def test_snapshot_owns_model_optimizer_scheduler_and_rng_storage():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    config = {
        "model": {"target": "torch.nn:Linear", "params": {}},
        "optimizer": {"target": "torch.optim:Adam", "params": {}},
        "scheduler": {"target": "torch.optim.lr_scheduler:StepLR", "params": {"step_size": 1}},
        "seed": 42,
    }
    snapshot = snapshot_checkpoint(model, optimizer, scheduler, 1, 0.5, config=config)
    weight = snapshot["model_state_dict"]["weight"].clone()
    moment = snapshot["optimizer_state_dict"]["state"][0]["exp_avg"].clone()
    scheduler_epoch = snapshot["scheduler_state_dict"]["last_epoch"]
    with torch.no_grad():
        model.weight.add_(20)
        optimizer.state[model.weight]["exp_avg"].add_(20)
    scheduler.step()
    assert torch.equal(snapshot["model_state_dict"]["weight"], weight)
    assert torch.equal(snapshot["optimizer_state_dict"]["state"][0]["exp_avg"], moment)
    assert snapshot["scheduler_state_dict"]["last_epoch"] == scheduler_epoch
