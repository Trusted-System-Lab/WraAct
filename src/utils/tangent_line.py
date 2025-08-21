"""
This module provides the algorithms to numerically calculate the tangent lines of
Sigmoid and Tanh functions.
There are three different implementations for the tangent line calculation: Python
float, Numpy, and PyTorch.

.. seealso::

    The numerical calculation of tangent lines of Sigmoid and Tanh functions is
    based on the following paper:

    - `Efficient Neural Network Verification via Adaptive Refinement and Adversarial
      Search
      <https://ecai2020.eu/papers/384_paper.pdf>`__
      :cite:`henriksen_efficient_2020`

"""

__docformat__ = "restructuredtext"
__all__ = ["get_parallel_tangent_line", "get_second_tangent_line"]

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from numpy import ndarray
from torch import Tensor

from .exceptions import NotConverged
from .functions import sigmoid, tanh

_LOG_MIN = 1e-6
_MAX_ITER = 100
_CONVERGE_TOL = 1e-4
_LOG_MIN_TENSOR = torch.tensor(_LOG_MIN, dtype=torch.float64)
_ZERO_TENSOR = torch.tensor(0.0, dtype=torch.float64)

# Disable the logging of Numba, which may be conflict with our logging.
logging.getLogger("numba").setLevel(logging.CRITICAL)


@njit
def _get_parallel_tangent_line_sigmoid_np(
    k: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:

    sign = 1.0 if get_big else -1.0

    temp = np.maximum(1.0 - 4.0 * k, 0.0)  # Avoid minimal negative value
    sigma = 2.0 * np.reciprocal(1.0 + sign * np.sqrt(temp))
    temp = np.maximum(sigma - 1.0, _LOG_MIN)
    x = -np.log(temp)
    # b = sigmoid(x) - k * x
    b = np.reciprocal(1.0 + np.exp(-x)) - k * x

    return b, k, x


k_np = np.random.rand(10) / 2
get_big_np = True
_get_parallel_tangent_line_sigmoid_np(k_np, get_big_np)


def _get_parallel_tangent_line_sigmoid_torch(
    k: Tensor, get_big: Tensor
) -> tuple[Tensor, Tensor, Tensor]:

    sign = 1.0 if get_big else -1.0

    temp = torch.maximum(1.0 - 4.0 * k, _ZERO_TENSOR)  # Avoid minimal negative value
    sigma = 2.0 * torch.reciprocal(1.0 + sign * torch.sqrt(temp))
    temp = torch.maximum(sigma - 1.0, _LOG_MIN_TENSOR)
    x = -torch.log(temp)

    b = sigmoid(x) - k * x

    return b, k, x


@njit
def _get_parallel_tangent_line_tanh_np(
    k: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:

    sign = 1.0 if get_big else -1.0
    temp = np.maximum(1.0 - k, 0.0)  # Avoid minimal negative value
    sigma = sign * np.sqrt(temp)
    x = np.log((1.0 + sigma) / (1.0 - sigma)) * 0.5
    b = np.tanh(x) - k * x

    return b, k, x


_get_parallel_tangent_line_tanh_np(k_np, get_big_np)


def _get_parallel_tangent_line_tanh_torch(
    k: Tensor, get_big: Tensor
) -> tuple[Tensor, Tensor, Tensor]:

    sign = 1.0 if get_big else -1.0
    temp = torch.maximum(1.0 - k, _ZERO_TENSOR)  # Avoid minimal negative value
    sigma = sign * torch.sqrt(temp)
    x = torch.log((1.0 + sigma) / (1.0 - sigma)) * 0.5
    b = tanh(x) - k * x

    return b, k, x


def _get_second_tanget_line_sigmoid_np(
    x1: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    x2 = np.zeros_like(x1, dtype=np.float64)
    y1 = np.reciprocal(1.0 + np.exp(-x1))

    for _ in range(_MAX_ITER):
        y2 = np.reciprocal(1.0 + np.exp(-x2))
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = _get_parallel_tangent_line_sigmoid_np(k, get_big)

        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConverged()


def _get_second_tangent_line_tanh_np(
    x1: ndarray | float, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    x2 = np.zeros_like(x1, dtype=np.float64)
    y1 = np.tanh(x1)

    for _ in range(_MAX_ITER):
        y2 = np.tanh(x2)
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = _get_parallel_tangent_line_tanh_np(k, get_big)

        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConverged()


def _get_second_tanget_line_sigmoid_torch(
    x1: Tensor, get_big: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    x2 = torch.zeros_like(x1, dtype=x1.dtype, device=x1.device)
    y1 = F.sigmoid(x1)

    for _ in range(_MAX_ITER):
        y2 = F.sigmoid(x2)
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = _get_parallel_tangent_line_sigmoid_torch(k, get_big)

        if torch.all(torch.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConverged()


def _get_second_tangent_line_tanh_torch(
    x1: Tensor, get_big: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    x2 = torch.zeros_like(x1, dtype=x1.dtype, device=x1.device)
    y1 = F.tanh(x1)

    for _ in range(_MAX_ITER):
        y2 = F.tanh(x2)
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = _get_parallel_tangent_line_tanh_torch(k, get_big)

        if torch.all(torch.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConverged()


def get_parallel_tangent_line(
    k: float | ndarray | Tensor,
    get_big: bool,
    func: Literal["sigmoid", "tanh"],
) -> tuple[
    float | ndarray | Tensor,
    float | ndarray | Tensor,
    float | ndarray | Tensor,
]:
    """
    Get the tangent line with a given slope :math:`k` and the tangent point.

    :param k: The slope of the tangent line.
    :param get_big: A boolean value to determine the choice of the tangent line based
        on the tangent point.
    :param func: The activation function to calculate the tangent line.

    :return: The intercepts, slope, and tangent point of the tangent line.
    """

    if isinstance(k, Tensor):
        k = k.to(dtype=torch.float64, device=k.device)
        get_big = torch.tensor(get_big, dtype=torch.bool, device=k.device)
        if func == "sigmoid":
            b, k, x = _get_parallel_tangent_line_sigmoid_torch(k, get_big)
        else:
            b, k, x = _get_parallel_tangent_line_tanh_torch(k, get_big)
        dtype = k.dtype
        b = b.to(dtype=dtype)
        k = k.to(dtype=dtype)
        x = x.to(dtype=dtype)

        return b, k, x

    elif isinstance(k, np.ndarray):
        k = k.astype(np.float64)

        if func == "sigmoid":
            b, k, x = _get_parallel_tangent_line_sigmoid_np(k, get_big)
        else:
            b, k, x = _get_parallel_tangent_line_tanh_np(k, get_big)
        dtype = k.dtype
        b = b.astype(dtype)
        k = k.astype(dtype)
        x = x.astype(dtype)

        return b, k, x

    else:
        k = np.array(k, dtype=np.float64)

        if func == "sigmoid":
            b, k, x = _get_parallel_tangent_line_sigmoid_np(k, get_big)
        else:
            b, k, x = _get_parallel_tangent_line_tanh_np(k, get_big)

        return float(b), float(k), float(x)


def get_second_tangent_line(
    x: float | ndarray | Tensor,
    get_big: bool,
    func: Literal["sigmoid", "tanh"],
) -> tuple[
    float | ndarray | Tensor,
    float | ndarray | Tensor,
    float | ndarray | Tensor,
]:
    """
    Get the second tangent line crossing a given point :math:`x` such that the tangent
    line does not take :math:`x` as the tangent point.

    :param x: The point where the tangent line crosses.
    :param get_big: Whether the tangent point is larger or smaller than the given point.
    :param func: The function to calculate the tangent line.

    :return: The intercepts, slope, and the tangent point of the tangent line.
    """

    if isinstance(x, Tensor):
        dtype = x.dtype
        x = x.to(dtype=torch.float64, device=x.device)
        get_big = torch.tensor(get_big, dtype=torch.bool, device=x.device)
        if func == "sigmoid":
            b, k, x = _get_second_tanget_line_sigmoid_torch(x, get_big)
        else:
            b, k, x = _get_second_tangent_line_tanh_torch(x, get_big)

        b = b.to(dtype=dtype)
        k = k.to(dtype=dtype)
        x = x.to(dtype=dtype)

        return b, k, x

    elif isinstance(x, np.ndarray):
        dtype = x.dtype
        x = x.astype(np.float64)

        if func == "sigmoid":
            b, k, x = _get_second_tanget_line_sigmoid_np(x, get_big)
        else:
            b, k, x = _get_second_tangent_line_tanh_np(x, get_big)

        b = b.astype(dtype)
        k = k.astype(dtype)
        x = x.astype(dtype)
        return b, k, x

    else:

        x = np.array(x, dtype=np.float64)

        if func == "sigmoid":
            b, k, x = _get_second_tanget_line_sigmoid_np(x, get_big)
        else:
            b, k, x = _get_second_tangent_line_tanh_np(x, get_big)

        return float(b), float(k), float(x)


# Warm up the functions to avoid the first call overhead
get_parallel_tangent_line(0.5, True, "sigmoid")
get_second_tangent_line(0.5, True, "sigmoid")
get_parallel_tangent_line(0.5, True, "tanh")
get_second_tangent_line(0.5, True, "tanh")
