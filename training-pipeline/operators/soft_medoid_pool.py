"""Differentiable medoid-like pooling for two-dimensional feature maps."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
        if len(pair) != 2:
            raise ValueError("expected an integer or a pair of integers")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError("pooling dimensions must be greater than zero")
    return pair


class SoftMedoidPool2d(nn.Module):
    """Pool local feature vectors using their soft medoid.

    Positions near the center of the local feature distribution receive more
    weight. Distances are root-mean-square distances across channels so one
    temperature works consistently for feature maps with different widths.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(kernel_size if stride is None else stride)
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("SoftMedoidPool2d expects NCHW input")

        batch_size, channels, height, width = features.shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        output_height = (height - kernel_height) // stride_height + 1
        output_width = (width - kernel_width) // stride_width + 1

        if output_height <= 0 or output_width <= 0:
            raise ValueError("kernel_size cannot exceed the input dimensions")

        window_size = kernel_height * kernel_width
        windows = F.unfold(
            features,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        windows = windows.reshape(
            batch_size, channels, window_size, -1
        ).permute(0, 3, 2, 1)

        squared_norms = windows.square().mean(dim=-1, keepdim=True)
        inner_products = torch.matmul(windows, windows.transpose(-1, -2))
        squared_distances = (
            squared_norms
            + squared_norms.transpose(-1, -2)
            - 2 * inner_products / channels
        ).clamp_min(0)
        distances = (squared_distances + self.eps).sqrt()
        centrality = distances.mean(dim=-1)
        weights = torch.softmax(-centrality / self.temperature, dim=-1)

        pooled = (windows * weights.unsqueeze(-1)).sum(dim=-2)
        return pooled.transpose(1, 2).reshape(
            batch_size, channels, output_height, output_width
        )
