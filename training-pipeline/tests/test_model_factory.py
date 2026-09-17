import pytest
from torch import nn

from models import (
    MODEL_NAMES,
    SimpleCNN,
    SimpleCNNWithSoftMedoid,
    create_model,
)


@pytest.mark.parametrize(
    ("name", "model_type"),
    (
        ("simple_cnn", SimpleCNN),
        ("simple_cnn_soft_medoid", SimpleCNNWithSoftMedoid),
    ),
)
def test_create_model_builds_registered_model(name, model_type) -> None:
    model = create_model(name, num_classes=3)

    assert isinstance(model, model_type)
    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == 3


def test_model_names_lists_registered_models() -> None:
    assert MODEL_NAMES == ("simple_cnn", "simple_cnn_soft_medoid")


def test_create_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model 'missing'"):
        create_model("missing")
