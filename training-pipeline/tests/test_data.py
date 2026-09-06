from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from cifar10_classifier.data import build_transforms


def test_only_training_transform_uses_random_augmentation() -> None:
    train_transform, eval_transform = build_transforms()
    assert any(isinstance(item, RandomCrop) for item in train_transform.transforms)
    assert any(isinstance(item, RandomHorizontalFlip) for item in train_transform.transforms)
    assert not any(isinstance(item, RandomCrop) for item in eval_transform.transforms)
    assert not any(isinstance(item, RandomHorizontalFlip) for item in eval_transform.transforms)
