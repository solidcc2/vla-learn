"""Resnet-18 model."""

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, branch: nn.Module, shortcut: nn.Module | None = None) -> None:
        super().__init__()
        self.branch = branch
        self.shortcut = shortcut if shortcut is not None else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.branch(x) + self.shortcut(x))


class ResNet18(nn.Module):

    def __init__(self, num_classes: int = 10, zero_init_residual: bool = True) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero")
        
        self.features = nn.Sequential(
            # stem
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # stage 1
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                )
            ),
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                )
            ),
            # stage 2
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                shortcut=nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=2, bias=False),
                    nn.BatchNorm2d(128)
                )
            ),
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                )
            ),
            # stage 3
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                ),
                shortcut=nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=2, bias=False),
                    nn.BatchNorm2d(256)
                )
            ),
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                )
            ),
            # stage 4
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                ),
                shortcut=nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=2, bias=False),
                    nn.BatchNorm2d(512)
                )
            ),
            ResidualBlock(
                branch=nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                )
            )
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, num_classes)
        )

        self._initialize_weights(zero_init_residual)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, ResidualBlock):
                    final_norm = [
                        child for child in module.branch.modules()
                        if isinstance(child, nn.BatchNorm2d)
                    ][-1]
                    nn.init.zeros_(final_norm.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
