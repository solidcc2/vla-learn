"""The convolutional neural network used by the baseline."""

from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """A compact CNN designed for 3 x 32 x 32 CIFAR images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero")

        self.features = nn.Sequential(
            # 3 x 32 x 32 -> 32 x 16 x 16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 32 x 16 x 16 -> 64 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 64 x 8 x 8 -> 128 x 4 x 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))
