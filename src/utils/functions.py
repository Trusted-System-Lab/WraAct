"""
This module implements the activation functions and their derivatives under different
data types, including Python float, Numpy, and PyTorch.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "relu",
    "drelu",
    "sigmoid",
    "dsigmoid",
    "tanh",
    "dtanh",
    "elu",
    "delu",
    "leakyrelu",
    "dleakyrelu",
    "silu",
    "dsilu",
]

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor


def relu_np(x: ndarray | float) -> ndarray | float:
    return np.maximum(x, 0.0)


def relu_torch(x: Tensor) -> Tensor:
    return F.relu(x)


def relu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return relu_torch(x)
    # return np.maximum(x, 0.0)
    return relu_np(x)


def drelu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, 0.0)


def drelu_torch(x: Tensor) -> Tensor:
    return (x > 0).float()


def drelu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return drelu_torch(x)
    return drelu_np(x)


def sigmoid_np(x: ndarray | float) -> ndarray | float:
    return np.reciprocal(1.0 + np.exp(-x))


def sigmoid_torch(x: Tensor) -> Tensor:
    return F.sigmoid(x)


def sigmoid(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return sigmoid_torch(x)
    return sigmoid_np(x)


def dsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s)


def dsigmoid_torch(x: Tensor) -> Tensor:
    s = F.sigmoid(x)
    return s * (1.0 - s)


def dsigmoid(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dsigmoid_torch(x)
    return dsigmoid_np(x)


def tanh_np(x: ndarray | float) -> ndarray | float:
    return np.tanh(x)


def tanh_torch(x: Tensor) -> Tensor:
    return F.tanh(x)


def tanh(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return tanh_torch(x)
    return tanh_np(x)


def dtanh_np(x: ndarray | float) -> ndarray | float:
    return 1.0 - np.tanh(x) ** 2


def dtanh_torch(x: Tensor) -> Tensor:
    return 1.0 - F.tanh(x) ** 2


def dtanh(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dtanh_torch(x)
    return dtanh_np(x)


def elu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, x, np.exp(x) - 1.0)


def elu_torch(x: Tensor) -> Tensor:
    return F.elu(x)


def elu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return elu_torch(x)
    return elu_np(x)


def delu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, np.exp(x))


def delu_torch(x: Tensor) -> Tensor:
    return torch.where(x > 0, 1.0, torch.exp(x))


def delu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return delu_torch(x)
    return delu_np(x)


def leakyrelu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, x, 0.01 * x)


def leakyrelu_torch(x: Tensor) -> Tensor:
    return F.leaky_relu(x)


def leakyrelu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return leakyrelu_torch(x)
    return leakyrelu_np(x)


def dleakyrelu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, 0.01)


def dleakyrelu_torch(x: Tensor) -> Tensor:
    return torch.where(x > 0, 1.0, 0.01)


def dleakyrelu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dleakyrelu_torch(x)
    return dleakyrelu_np(x)


def silu_np(x: ndarray | float) -> ndarray | float:
    return np.reciprocal(1.0 + np.exp(-x)) * x


def silu_torch(x: Tensor) -> Tensor:
    return F.silu(x)


def silu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return silu_torch(x)
    return silu_np(x)


def dsilu_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s + x * s * (1.0 - s)


def dsilu_torch(x: Tensor) -> Tensor:
    s = F.sigmoid(x)
    return s + x * s * (1.0 - s)


def dsilu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dsilu_torch(x)
    return dsilu_np(x)
