"""CIFAR-10 and CIFAR-100 train/validation/test data construction."""

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .base import DataModule


_DATASETS = {
    "cifar10": {
        "type": datasets.CIFAR10,
        "classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "type": datasets.CIFAR100,
        "classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


def stratified_split_indices(
    targets: list[int],
    validation_size: int,
    split_seed: int,
    num_classes: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic, class-balanced training and validation indices."""
    if validation_size <= 0 or validation_size >= len(targets):
        raise ValueError("validation_size must be between zero and the training size")
    if validation_size % num_classes:
        raise ValueError("validation_size must be divisible by the number of classes")
    if split_seed < 0:
        raise ValueError("split_seed must be nonnegative")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    if set(by_class) != set(range(num_classes)):
        raise ValueError("dataset targets do not contain the expected classes")

    per_class = validation_size // num_classes
    generator = torch.Generator().manual_seed(split_seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index in range(num_classes):
        indices = by_class[class_index]
        if per_class >= len(indices):
            raise ValueError("validation split would empty a training class")
        order = torch.randperm(len(indices), generator=generator).tolist()
        validation_indices.extend(indices[position] for position in order[:per_class])
        train_indices.extend(indices[position] for position in order[per_class:])
    return sorted(train_indices), sorted(validation_indices)


class CIFARDataModule(DataModule):
    def __init__(
        self,
        dataset: str = "cifar10",
        validation_size: int = 5000,
        split_seed: int = 42,
    ) -> None:
        if dataset not in _DATASETS:
            raise ValueError(f"Unsupported CIFAR dataset: {dataset}")
        self.dataset = dataset
        self.validation_size = validation_size
        self.split_seed = split_seed

    def _transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        specification = _DATASETS[self.dataset]
        normalize = transforms.Normalize(specification["mean"], specification["std"])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        evaluation_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return train_transform, evaluation_transform

    def create_train_loaders(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 0,
        download: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        specification = _DATASETS[self.dataset]
        dataset_type = specification["type"]
        train_transform, evaluation_transform = self._transforms()
        training_data = dataset_type(
            root=data_dir, train=True, transform=train_transform, download=download,
        )
        validation_data = dataset_type(
            root=data_dir, train=True, transform=evaluation_transform, download=False,
        )
        train_indices, validation_indices = stratified_split_indices(
            training_data.targets,
            self.validation_size,
            self.split_seed,
            specification["classes"],
        )
        return (
            DataLoader(
                Subset(training_data, train_indices), batch_size=batch_size,
                shuffle=True, num_workers=num_workers,
            ),
            DataLoader(
                Subset(validation_data, validation_indices), batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
            ),
        )

    def create_test_loader(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 0,
        download: bool = True,
    ) -> DataLoader:
        specification = _DATASETS[self.dataset]
        _, evaluation_transform = self._transforms()
        test_data = specification["type"](
            root=data_dir, train=False, transform=evaluation_transform, download=download,
        )
        return DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
