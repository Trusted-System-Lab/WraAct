__docformat__ = ["restructuredtext"]
__all__ = ["cal_minimum_point"]

import torch
from torch import Tensor


def _cal_minimum_point_1d(A: Tensor, l: Tensor, u: Tensor) -> Tensor:
    A_p = A > 0
    A_n = A < 0
    return A_p * l + A_n * u


def _cal_minimum_point_3d(A: Tensor, l: Tensor, u: Tensor) -> Tensor:
    A_p = A > 0
    A_n = A < 0
    return A_p * l + A_n * u


_example_inputs1_f32 = (
    torch.rand((2, 3), dtype=torch.float32),
    torch.rand((3,), dtype=torch.float32),
    torch.rand((3,), dtype=torch.float32),
)
_example_inputs2_f32 = (
    torch.rand((2, 3, 4, 5), dtype=torch.float32),
    torch.rand((3, 4, 5), dtype=torch.float32),
    torch.rand((3, 4, 5), dtype=torch.float32),
)

_cal_minimum_point_1d_f32 = torch.jit.trace(_cal_minimum_point_1d, _example_inputs1_f32)
_cal_minimum_point_3d_f32 = torch.jit.trace(_cal_minimum_point_3d, _example_inputs2_f32)

_example_inputs1_f64 = (
    torch.rand((2, 3), dtype=torch.float64),
    torch.rand((3,), dtype=torch.float64),
    torch.rand((3,), dtype=torch.float64),
)
_example_inputs2_f64 = (
    torch.rand((2, 3, 4, 5), dtype=torch.float64),
    torch.rand((3, 4, 5), dtype=torch.float64),
    torch.rand((3, 4, 5), dtype=torch.float64),
)

_cal_minimum_point_1d_f64 = torch.jit.trace(_cal_minimum_point_1d, _example_inputs1_f64)
_cal_minimum_point_3d_f64 = torch.jit.trace(_cal_minimum_point_3d, _example_inputs2_f64)


def cal_minimum_point(A: Tensor, l: Tensor, u: Tensor) -> Tensor:
    """
    Calculate the point to make the linear relaxation minimum for lower bound.

    :param A: The matrix of the linear relaxation.
    :param l: The scalar lower bound.
    :param u: The scalar upper bound.
    """
    d = A.dim()
    dtype = A.dtype

    if dtype == torch.float32:
        if d == 2:
            return _cal_minimum_point_1d_f32(A, l, u)
        elif d == 4:
            return _cal_minimum_point_3d_f32(A, l, u)
        else:
            raise ValueError(
                f"Unsupported dimension: " f"A: {A.shape}, l: {l.shape}, u: {u.shape}."
            )
    elif dtype == torch.float64:
        if d == 2:
            return _cal_minimum_point_1d_f64(A, l, u)
        elif d == 4:
            return _cal_minimum_point_3d_f64(A, l, u)
        else:
            raise ValueError(
                f"Unsupported dimension: " f"A: {A.shape}, l: {l.shape}, u: {u.shape}."
            )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}.")
