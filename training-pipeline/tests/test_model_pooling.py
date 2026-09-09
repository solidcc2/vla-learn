from model import SimpleCNN
from operators import SoftMedoidPool2d


def test_simple_cnn_uses_soft_medoid_for_each_downsampling_stage() -> None:
    model = SimpleCNN()
    pooling_layers = [
        layer for layer in model.features if isinstance(layer, SoftMedoidPool2d)
    ]
    assert len(pooling_layers) == 3
