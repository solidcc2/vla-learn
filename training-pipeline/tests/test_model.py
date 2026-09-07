import pytest
import torch

from model import SimpleCNN


def test_simple_cnn_returns_one_logit_per_class() -> None:
    model = SimpleCNN(num_classes=10)
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)


def test_simple_cnn_rejects_invalid_class_count() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        SimpleCNN(num_classes=0)
