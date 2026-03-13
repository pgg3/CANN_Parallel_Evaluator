"""fn 格式示例：ELU 算子（带 init_params）

module_fn 定义算子逻辑，Model 代理调用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return F.elu(x, alpha=alpha)


class Model(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.alpha)


def get_inputs():
    return [torch.randn(16, 16384)]


def get_init_inputs():
    return [1.0]
