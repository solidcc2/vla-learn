from importlib import import_module

import pytest
import torch


def soft_medoid_pool_type():
    try:
        module = import_module("operators")
    except ModuleNotFoundError:
        pytest.fail("the operators package containing SoftMedoidPool2d is missing")
    return module.SoftMedoidPool2d


def test_soft_medoid_pool_downsamples_spatial_dimensions() -> None:
    pool = soft_medoid_pool_type()(kernel_size=2)

    output = pool(torch.randn(2, 3, 4, 6))

    assert output.shape == (2, 3, 2, 3)


def test_soft_medoid_pool_preserves_constant_features() -> None:
    pool = soft_medoid_pool_type()(kernel_size=2)
    features = torch.full((1, 3, 4, 4), 7.0)

    output = pool(features)

    torch.testing.assert_close(output, torch.full((1, 3, 2, 2), 7.0))


def test_soft_medoid_pool_downweights_a_local_outlier() -> None:
    pool = soft_medoid_pool_type()(kernel_size=2, temperature=1.0)
    features = torch.tensor([[[[1.0, 2.0], [2.2, 10.0]]]])

    output = pool(features)

    assert output.item() == pytest.approx(1.9, abs=0.1)
    assert output.item() < features.mean().item()


def test_soft_medoid_pool_propagates_finite_gradients() -> None:
    pool = soft_medoid_pool_type()(kernel_size=2)
    features = torch.randn(2, 3, 4, 4, requires_grad=True)

    pool(features).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad) > 0


def test_soft_medoid_pool_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        soft_medoid_pool_type()(kernel_size=2, temperature=0.0)
