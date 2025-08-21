__docformat__ = ["restructuredtext"]
__all__ = ["cal_scalar_bound"]

import torch
from torch import Tensor


def _cal_scalar_bound_1d(A: Tensor, b: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1,)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = b + (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


def _cal_scalar_bound_without_bias_1d(A: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1,)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


def _cal_scalar_bound_3d(A: Tensor, b: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1, 2, 3)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = b + (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


def _cal_scalar_bound_without_bias_3d(A: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1, 2, 3)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


_A_f32 = torch.rand((2, 3), dtype=torch.float32)
_b_f32 = torch.rand((2,), dtype=torch.float32)
_s1_f32 = torch.rand((3,), dtype=torch.float32)
_s2_f32 = torch.rand((3,), dtype=torch.float32)

_A_3d_f32 = torch.rand((2, 3, 4, 5), dtype=torch.float32)
_b_3d_f32 = torch.rand((2,), dtype=torch.float32)
_s1_3d_f32 = torch.rand((3, 4, 5), dtype=torch.float32)
_s2_3d_f32 = torch.rand((3, 4, 5), dtype=torch.float32)

_A_f64 = torch.rand((2, 3), dtype=torch.float64)
_b_f64 = torch.rand((2,), dtype=torch.float64)
_s1_f64 = torch.rand((3,), dtype=torch.float64)
_s2_f64 = torch.rand((3,), dtype=torch.float64)

_A_3d_f64 = torch.rand((2, 3, 4, 5), dtype=torch.float64)
_b_3d_f64 = torch.rand((2,), dtype=torch.float64)
_s1_3d_f64 = torch.rand((3, 4, 5), dtype=torch.float64)
_s2_3d_f64 = torch.rand((3, 4, 5), dtype=torch.float64)


_example_inputs1_f32 = (_A_f32, _b_f32, _s1_f32, _s2_f32)
_example_inputs1_without_bias_f32 = (_A_f32, _s1_f32, _s2_f32)
_example_inputs2_f32 = (_A_3d_f32, _b_3d_f32, _s1_3d_f32, _s2_3d_f32)
_example_inputs2_without_bias_f32 = (_A_3d_f32, _s1_3d_f32, _s2_3d_f32)
_example_inputs1_f64 = (_A_f64, _b_f64, _s1_f64, _s2_f64)
_example_inputs1_without_bias_f64 = (_A_f64, _s1_f64, _s2_f64)
_example_inputs2_f64 = (_A_3d_f64, _b_3d_f64, _s1_3d_f64, _s2_3d_f64)
_example_inputs2_without_bias_f64 = (_A_3d_f64, _s1_3d_f64, _s2_3d_f64)

_cal_scalar_bound_1d_f32 = torch.jit.trace(_cal_scalar_bound_1d, _example_inputs1_f32)
_cal_scalar_bound_without_bias_1d_f32 = torch.jit.trace(
    _cal_scalar_bound_without_bias_1d, _example_inputs1_without_bias_f32
)
_cal_scalar_bound_3d_f32 = torch.jit.trace(_cal_scalar_bound_3d, _example_inputs2_f32)
_cal_scalar_bound_without_bias_3d_f32 = torch.jit.trace(
    _cal_scalar_bound_without_bias_3d, _example_inputs2_without_bias_f32
)


_cal_scalar_bound_1d_f64 = torch.jit.trace(_cal_scalar_bound_1d, _example_inputs1_f64)
_cal_scalar_bound_without_bias_1d_f64 = torch.jit.trace(
    _cal_scalar_bound_without_bias_1d, _example_inputs1_without_bias_f64
)
_cal_scalar_bound_3d_f64 = torch.jit.trace(_cal_scalar_bound_3d, _example_inputs2_f64)
_cal_scalar_bound_without_bias_3d_f64 = torch.jit.trace(
    _cal_scalar_bound_without_bias_3d, _example_inputs2_without_bias_f64
)


def cal_scalar_bound(A: Tensor, b: Tensor | None, s1: Tensor, s2: Tensor) -> Tensor:
    """
    Calculate the scalar bound of the linear relaxation.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param s1: The scalar lower or upper bound.
    :param s2: The scalar upper or lower bound.

    :return: The scalar bound of the linear relaxation.
    """

    d = A.dim()
    dtype = A.dtype

    if dtype == torch.float32:
        if b is not None:
            if d == 2:
                return _cal_scalar_bound_1d_f32(A, b, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_3d_f32(A, b, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape}, b: {b.shape}, l: {s1.shape}, u: {s2.shape}."
                )
        else:
            if d == 2:
                return _cal_scalar_bound_without_bias_1d_f32(A, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_without_bias_3d_f32(A, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape} l: {s1.shape}, u: {s2.shape}."
                )
    elif dtype == torch.float64:
        if b is not None:
            if d == 2:
                return _cal_scalar_bound_1d_f64(A, b, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_3d_f64(A, b, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape}, b: {b.shape}, l: {s1.shape}, u: {s2.shape}."
                )
        else:
            if d == 2:
                return _cal_scalar_bound_without_bias_1d_f64(A, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_without_bias_3d_f64(A, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape} l: {s1.shape}, u: {s2.shape}."
                )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}.")
