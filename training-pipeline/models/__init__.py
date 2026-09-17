"""Available CIFAR-10 models and their construction factory."""

from collections.abc import Callable

from torch import nn

from .simple_cnn import SimpleCNN
from .simple_cnn_soft_medoid import SimpleCNNWithSoftMedoid


ModelConstructor = Callable[[int], nn.Module]

_MODEL_CONSTRUCTORS: dict[str, ModelConstructor] = {
    "simple_cnn": SimpleCNN,
    "simple_cnn_soft_medoid": SimpleCNNWithSoftMedoid,
}

MODEL_NAMES = tuple(_MODEL_CONSTRUCTORS)


def create_model(name: str, num_classes: int = 10) -> nn.Module:
    """Create a model registered under ``name``."""
    try:
        constructor = _MODEL_CONSTRUCTORS[name]
    except KeyError as error:
        choices = ", ".join(MODEL_NAMES)
        raise ValueError(
            f"Unknown model {name!r}; expected one of: {choices}"
        ) from error
    return constructor(num_classes)


__all__ = (
    "MODEL_NAMES",
    "SimpleCNN",
    "SimpleCNNWithSoftMedoid",
    "create_model",
)
