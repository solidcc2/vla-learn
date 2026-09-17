"""Resnet-18 model."""

import torch
from torch import Tensor, nn

class ResidualBlock(nn.Module):
    def __init__(self, branch, shortcut = None):
        super().__init__()
        self.branch = branch
        self.shortcut = shortcut if shortcut is not None else nn.Identity()
    
    def forward(self, x):
        return torch.relu(self.branch(x) + self.shortcut(x))

class ResNet18(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
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

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
