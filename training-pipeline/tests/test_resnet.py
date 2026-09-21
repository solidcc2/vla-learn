import math

import torch
from torch import nn

from models.resnet import ResidualBlock, ResNet18


def residual_final_norms(model: ResNet18) -> list[nn.BatchNorm2d]:
    norms = []
    for module in model.modules():
        if isinstance(module, ResidualBlock):
            branch_norms = [
                child for child in module.branch.modules()
                if isinstance(child, nn.BatchNorm2d)
            ]
            norms.append(branch_norms[-1])
    return norms


def test_resnet18_initializes_convolution_norm_and_classifier() -> None:
    torch.manual_seed(7)
    model = ResNet18(num_classes=10, zero_init_residual=False)

    first_conv = next(
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    )
    expected_std = math.sqrt(2.0 / (first_conv.out_channels * 3 * 3))
    assert abs(first_conv.weight.std().item() - expected_std) < 0.01

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))

    classifier = model.classifier[-1]
    assert isinstance(classifier, nn.Linear)
    assert abs(classifier.weight.std().item() - 0.01) < 0.002
    assert torch.equal(classifier.bias, torch.zeros_like(classifier.bias))


def test_resnet18_can_zero_initialize_residual_branches() -> None:
    model = ResNet18(zero_init_residual=True)
    final_norms = residual_final_norms(model)
    assert len(final_norms) == 8
    for norm in final_norms:
        assert torch.equal(norm.weight, torch.zeros_like(norm.weight))


def test_resnet18_initialization_uses_the_training_seed() -> None:
    torch.manual_seed(42)
    first = ResNet18()
    torch.manual_seed(42)
    second = ResNet18()
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_parameter, second_parameter)
