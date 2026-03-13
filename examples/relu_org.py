"""org 格式示例：ReLU 算子

Model 类 + get_inputs / get_init_inputs
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def get_inputs():
    return [torch.randn(16, 16384)]


def get_init_inputs():
    return []
