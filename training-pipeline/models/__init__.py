"""Built-in CIFAR-10 model classes."""

from .resnet import ResNet18
from .simple_cnn import SimpleCNN
from .simple_cnn_soft_medoid import SimpleCNNWithSoftMedoid

__all__ = (
    "ResNet18",
    "SimpleCNN",
    "SimpleCNNWithSoftMedoid",
)
