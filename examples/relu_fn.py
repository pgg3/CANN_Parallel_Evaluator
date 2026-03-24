"""fn 格式示例：ReLU 算子

module_fn 定义算子逻辑，Model 代理调用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x)


def get_inputs():
    return [torch.randn(24, 1024)]


def get_init_inputs():
    return []
