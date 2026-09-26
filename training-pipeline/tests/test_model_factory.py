import pytest
import torch
from torch import nn

from components import create_data_module, create_model, create_optimizer, create_scheduler, resolve_target
from models.resnet import ResNet18
from data_modules.cifar import CIFARDataModule


def test_create_model_loads_resnet_from_python_target() -> None:
    model = create_model({
        "target": "models.resnet:ResNet18",
        "params": {"num_classes": 3},
    })
    assert isinstance(model, ResNet18)
    assert model.classifier[-1].out_features == 3

def test_resnet18_forward_has_expected_shape() -> None:
    model = ResNet18(num_classes=10).eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (2, 10)


@pytest.mark.parametrize("target", ["missing", "models.missing:Model", "models.simple_cnn:Missing"])
def test_resolve_target_rejects_invalid_or_missing_target(target) -> None:
    with pytest.raises(ValueError):
        resolve_target(target)


def test_create_model_rejects_non_module_class() -> None:
    with pytest.raises(TypeError, match="nn.Module"):
        create_model({"target": "builtins:dict", "params": {}})


def test_create_data_module_loads_cifar_from_python_target() -> None:
    data_module = create_data_module({
        "target": "data_modules.cifar:CIFARDataModule",
        "params": {"dataset": "cifar100", "validation_size": 5000},
    })
    assert isinstance(data_module, CIFARDataModule)
    assert data_module.dataset == "cifar100"


def test_create_data_module_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="DataModule"):
        create_data_module({"target": "builtins:dict", "params": {}})


def test_optimizer_and_cosine_scheduler_are_configured() -> None:
    model = nn.Linear(3, 2)
    optimizer = create_optimizer({
        "target": "torch.optim:SGD",
        "params": {"lr": 0.1, "momentum": 0.9},
    }, model.parameters())
    scheduler = create_scheduler({
        "target": "torch.optim.lr_scheduler:CosineAnnealingLR",
        "params": {"T_max": 2, "eta_min": 0.01},
    }, optimizer)
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert create_scheduler(None, optimizer) is None


def test_metric_scheduler_is_rejected() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(TypeError, match="requires arguments"):
        create_scheduler({
            "target": "torch.optim.lr_scheduler:ReduceLROnPlateau",
            "params": {},
        }, optimizer)
