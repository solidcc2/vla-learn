from collections import Counter

from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from data_modules.cifar import CIFARDataModule, stratified_split_indices


def test_only_training_transform_uses_random_augmentation() -> None:
    train_transform, eval_transform = CIFARDataModule()._transforms()
    assert any(isinstance(item, RandomCrop) for item in train_transform.transforms)
    assert any(isinstance(item, RandomHorizontalFlip) for item in train_transform.transforms)
    assert not any(isinstance(item, RandomCrop) for item in eval_transform.transforms)
    assert not any(isinstance(item, RandomHorizontalFlip) for item in eval_transform.transforms)


def test_stratified_split_is_balanced_disjoint_complete_and_repeatable() -> None:
    targets = [class_index for class_index in range(10) for _ in range(5000)]
    first = stratified_split_indices(targets, 5000, split_seed=42, num_classes=10)
    second = stratified_split_indices(targets, 5000, split_seed=42, num_classes=10)
    train_indices, validation_indices = first

    assert first == second
    assert len(train_indices) == 45_000
    assert len(validation_indices) == 5_000
    assert set(train_indices).isdisjoint(validation_indices)
    assert sorted(train_indices + validation_indices) == list(range(len(targets)))
    assert Counter(targets[index] for index in validation_indices) == {
        class_index: 500 for class_index in range(10)
    }


def test_split_seed_changes_validation_membership() -> None:
    targets = [class_index for class_index in range(10) for _ in range(10)]
    _, first = stratified_split_indices(targets, 20, split_seed=1, num_classes=10)
    _, second = stratified_split_indices(targets, 20, split_seed=2, num_classes=10)
    assert first != second


def test_cifar_module_supports_cifar100() -> None:
    module = CIFARDataModule(
        dataset="cifar100", validation_size=5000, split_seed=42,
    )
    assert module.dataset == "cifar100"
