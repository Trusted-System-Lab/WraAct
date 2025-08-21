"""
This module contains the implementation of the specialized ad-hoc back-substitution for
the ReLU activation function.
"""

__docformat__ = ["restructuredtext"]
__all__ = ["back_substitute_relu"]

import torch
from torch import Tensor


def _back_substitute_relu_1d_lower(
    A: Tensor, b: Tensor, sl: Tensor, su: Tensor, tu: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    b = b + (An * tu).sum(dim=1)
    A = Ap * sl + An * su

    return A, b


def _back_substitute_relu_1d_upper(
    A: Tensor, b: Tensor, sl: Tensor, su: Tensor, tu: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    b = b + (Ap * tu).sum(dim=1)
    A = Ap * su + An * sl

    return A, b


def _back_substitute_relu_without_bias_1d_lower(
    A: Tensor, sl: Tensor, su: Tensor, tu: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    b = (An * tu).sum(dim=1)
    A = Ap * sl + An * su

    return A, b


def _back_substitute_relu_without_bias_1d_upper(
    A: Tensor, sl: Tensor, su: Tensor, tu: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    b = (Ap * tu).sum(dim=1)
    A = Ap * su + An * sl

    return A, b


_A_f32 = torch.rand((2, 3), dtype=torch.float32)
_b_f32 = torch.rand((2,), dtype=torch.float32)
_s1_f32 = torch.rand((3,), dtype=torch.float32)
_s2_f32 = torch.rand((3,), dtype=torch.float32)
_s1_2d_f32 = torch.rand((3, 3), dtype=torch.float32)
_s2_2d_f32 = torch.rand((3, 3), dtype=torch.float32)
_t1_f32 = torch.rand((3,), dtype=torch.float32)

_A_3d_f32 = torch.rand((2, 3, 4), dtype=torch.float32)
_b_3d_f32 = torch.rand((2, 3), dtype=torch.float32)
_s1_3d_f32 = torch.rand((2, 4, 4), dtype=torch.float32)
_s2_3d_f32 = torch.rand((2, 4, 4), dtype=torch.float32)
_t1_3d_f32 = torch.rand((2, 4), dtype=torch.float32)


_A_f64 = torch.rand((2, 3), dtype=torch.float64)
_b_f64 = torch.rand((2,), dtype=torch.float64)
_s1_f64 = torch.rand((3,), dtype=torch.float64)
_s2_f64 = torch.rand((3,), dtype=torch.float64)
_s1_2d_f64 = torch.rand((3, 3), dtype=torch.float64)
_s2_2d_f64 = torch.rand((3, 3), dtype=torch.float64)
_t1_f64 = torch.rand((3,), dtype=torch.float64)

_A_3d_f64 = torch.rand((2, 3, 4), dtype=torch.float64)
_b_3d_f64 = torch.rand((2, 3), dtype=torch.float64)
_s1_3d_f64 = torch.rand((2, 4, 4), dtype=torch.float64)
_s2_3d_f64 = torch.rand((2, 4, 4), dtype=torch.float64)
_t1_3d_f64 = torch.rand((2, 4), dtype=torch.float64)


_example_input_1d_f32 = (_A_f32, _b_f32, _s1_f32, _s2_f32, _t1_f32)
_example_input_without_bias_1d_f32 = (_A_f32, _s1_f32, _s2_f32, _t1_f32)
_example_input_2d_f32 = (_A_f32, _b_f32, _s1_2d_f32, _s2_2d_f32, _t1_f32)
_example_input_without_bias_2d_f32 = (_A_f32, _s1_2d_f32, _s2_2d_f32, _t1_f32)
_example_input_3d_f32 = (
    _A_3d_f32,
    _b_3d_f32,
    _s1_3d_f32,
    _s2_3d_f32,
    _t1_3d_f32,
)
_example_input_without_bias_3d_f32 = (
    _A_3d_f32,
    _s1_3d_f32,
    _s2_3d_f32,
    _t1_3d_f32,
)

_example_input_1d_f64 = (_A_f64, _b_f64, _s1_f64, _s2_f64, _t1_f64)
_example_input_without_bias_1d_f64 = (_A_f64, _s1_f64, _s2_f64, _t1_f64)
_example_input_2d_f64 = (_A_f64, _b_f64, _s1_2d_f64, _s2_2d_f64, _t1_f64)
_example_input_without_bias_2d_f64 = (_A_f64, _s1_2d_f64, _s2_2d_f64, _t1_f64)
_example_input_3d_f64 = (
    _A_3d_f64,
    _b_3d_f64,
    _s1_3d_f64,
    _s2_3d_f64,
    _t1_3d_f64,
)
_example_input_without_bias_3d_f64 = (
    _A_3d_f64,
    _s1_3d_f64,
    _s2_3d_f64,
    _t1_3d_f64,
)


_back_substitute_relu_1d_lower_f32 = torch.jit.trace(
    _back_substitute_relu_1d_lower, _example_input_1d_f32
)
_back_substitute_relu_1d_upper_f32 = torch.jit.trace(
    _back_substitute_relu_1d_upper, _example_input_1d_f32
)
_back_substitute_relu_without_bias_1d_lower_f32 = torch.jit.trace(
    _back_substitute_relu_without_bias_1d_lower, _example_input_without_bias_1d_f32
)
_back_substitute_relu_without_bias_1d_upper_f32 = torch.jit.trace(
    _back_substitute_relu_without_bias_1d_upper, _example_input_without_bias_1d_f32
)

_back_substitute_relu_1d_lower_f64 = torch.jit.trace(
    _back_substitute_relu_1d_lower, _example_input_1d_f64
)
_back_substitute_relu_1d_upper_f64 = torch.jit.trace(
    _back_substitute_relu_1d_upper, _example_input_1d_f64
)
_back_substitute_relu_without_bias_1d_lower_f64 = torch.jit.trace(
    _back_substitute_relu_without_bias_1d_lower, _example_input_without_bias_1d_f64
)
_back_substitute_relu_without_bias_1d_upper_f64 = torch.jit.trace(
    _back_substitute_relu_without_bias_1d_upper, _example_input_without_bias_1d_f64
)


def back_substitute_relu(
    A: Tensor,
    b: Tensor | None,
    sl: Tensor,
    su: Tensor,
    tu: Tensor,
    is_lower: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a non-linear operation. Based on
    calculating the lower or upper linear relaxation, lower or upper slopes and
    intercepts are used.

    .. tip::
        This is a specialized implementation for the ReLU activation function.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param sl: The slopes of the lower linear relaxation of the non-linear operation.
    :param su: The slopes of the upper linear relaxation of the non-linear operation.
    :param tu: The intercepts of the upper linear relaxation of the non-linear
        operation.
    :param is_lower: If True, calculate the lower relaxation; otherwise, calculate the
        upper relaxation.

    :return: The matrix and bias of the linear relaxation.
    """
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"The data type {A.dtype} is not supported.")

    if is_lower:
        if sl.dim() == su.dim() == 1 and tu.dim() == 1:

            if b is not None:
                if A.dtype == torch.float32:
                    return _back_substitute_relu_1d_lower_f32(A, b, sl, su, tu)
                return _back_substitute_relu_1d_lower_f64(A, b, sl, su, tu)
            if A.dtype == torch.float32:
                return _back_substitute_relu_without_bias_1d_lower_f32(A, sl, su, tu)
            return _back_substitute_relu_without_bias_1d_lower_f64(A, sl, su, tu)

        else:
            raise ValueError(
                f"The dimensions are not supported. "
                f"A: {A.shape}, b: {b.shape}, "
                f"sl: {sl.shape}, su: {su.shape}, "
                f"tu: {tu.shape}."
            )

    if sl.dim() == su.dim() == 1 and tu.dim() == 1:

        if b is not None:
            if A.dtype == torch.float32:
                return _back_substitute_relu_1d_upper_f32(A, b, sl, su, tu)
            return _back_substitute_relu_1d_upper_f64(A, b, sl, su, tu)
        if A.dtype == torch.float32:
            return _back_substitute_relu_without_bias_1d_upper_f32(A, sl, su, tu)
        return _back_substitute_relu_without_bias_1d_upper_f64(A, sl, su, tu)

    else:
        raise ValueError(
            f"The dimensions are not supported. "
            f"A: {A.shape}, b: {b.shape}, "
            f"sl: {sl.shape}, su: {su.shape}, "
            f"tu: {tu.shape}."
        )
