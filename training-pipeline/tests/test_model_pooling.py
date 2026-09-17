from torch import nn

from models import SimpleCNN, SimpleCNNWithSoftMedoid
from operators import SoftMedoidPool2d


def test_simple_cnn_uses_max_pooling() -> None:
    assert sum(isinstance(layer, nn.MaxPool2d) for layer in SimpleCNN().features) == 3


def test_soft_medoid_variant_uses_soft_medoid_pooling() -> None:
    assert sum(
        isinstance(layer, SoftMedoidPool2d)
        for layer in SimpleCNNWithSoftMedoid().features
    ) == 3
