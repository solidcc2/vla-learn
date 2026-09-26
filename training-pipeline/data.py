"""Interfaces for dynamically configured training data modules."""

from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader


class DataModule(ABC):
    """Construct task-specific loaders while keeping runtime paths external."""

    @abstractmethod
    def create_train_loaders(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 0,
        download: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        """Return training and validation loaders."""

    @abstractmethod
    def create_test_loader(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 0,
        download: bool = True,
    ) -> DataLoader:
        """Return the held-out test loader."""
