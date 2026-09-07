import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from engine import evaluate, train_one_epoch


def _loader() -> DataLoader:
    images = torch.randn(8, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_evaluate_reports_metrics() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 2))
    metrics = evaluate(model, _loader(), nn.CrossEntropyLoss(), torch.device("cpu"))
    assert metrics.samples == 8
    assert metrics.loss >= 0
    assert 0 <= metrics.accuracy <= 1


def test_train_one_epoch_updates_model_parameters() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model[1].weight.detach().clone()
    metrics = train_one_epoch(
        model, _loader(), nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )
    assert metrics.samples == 8
    assert not torch.equal(before, model[1].weight)
