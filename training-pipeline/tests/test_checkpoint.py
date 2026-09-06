import torch
from torch import nn

from cifar10_classifier.checkpoint import load_checkpoint, save_checkpoint


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
