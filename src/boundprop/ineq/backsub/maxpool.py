__docformat__ = ["restructuredtext"]
__all__ = ["back_substitute_maxpool2d"]

import torch
import torch.nn.functional as F
from torch import Tensor


def _back_substitute_maxpool_naive(
    A: Tensor, b: Tensor, s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    d = (1, 2)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)

    b = b + torch.einsum("abc, bc->abc", Ap, t1).sum(dim=d)
    b = b + torch.einsum("abc, bc->abc", An, t2).sum(dim=d)
    A = torch.einsum("abc, bcd->abcd", Ap, s1)
    A = A + torch.einsum("abc, bcd->abcd", An, s2)

    return A, b


def _back_substitute_maxpool_without_bias_naive(
    A: Tensor, s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    d = (1, 2)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)

    b = torch.einsum("abc, bc->abc", Ap, t1).sum(dim=d)
    b = b + torch.einsum("abc, bc->abc", An, t2).sum(dim=d)
    A = torch.einsum("abc, bcd->abcd", Ap, s1)
    A = A + torch.einsum("abc, bcd->abcd", An, s2)

    return A, b


_A_f32 = torch.rand((2, 3, 4), dtype=torch.float32)
_b_f32 = torch.rand((2,), dtype=torch.float32)
_s1_f32 = torch.rand((3, 4, 5), dtype=torch.float32)
_s2_f32 = torch.rand((3, 4, 5), dtype=torch.float32)
_t1_f32 = torch.rand((3, 4), dtype=torch.float32)
_t2_f32 = torch.rand((3, 4), dtype=torch.float32)

_A_4d_f32 = torch.rand((2, 3, 4, 5), dtype=torch.float32)
_b_4d_f32 = torch.rand((2, 3), dtype=torch.float32)
_s1_4d_f32 = torch.rand((2, 4, 5, 2), dtype=torch.float32)
_s2_4d_f32 = torch.rand((2, 4, 5, 2), dtype=torch.float32)
_t1_4d_f32 = torch.rand((2, 4, 5), dtype=torch.float32)
_t2_4d_f32 = torch.rand((2, 4, 5), dtype=torch.float32)

_A_f64 = torch.rand((2, 3, 4), dtype=torch.float64)
_b_f64 = torch.rand((2,), dtype=torch.float64)
_s1_f64 = torch.rand((3, 4, 5), dtype=torch.float64)
_s2_f64 = torch.rand((3, 4, 5), dtype=torch.float64)
_t1_f64 = torch.rand((3, 4), dtype=torch.float64)
_t2_f64 = torch.rand((3, 4), dtype=torch.float64)

_A_4d_f64 = torch.rand((2, 3, 4, 5), dtype=torch.float64)
_b_4d_f64 = torch.rand((2, 3), dtype=torch.float64)
_s1_4d_f64 = torch.rand((2, 4, 5, 2), dtype=torch.float64)
_s2_4d_f64 = torch.rand((2, 4, 5, 2), dtype=torch.float64)
_t1_4d_f64 = torch.rand((2, 4, 5), dtype=torch.float64)
_t2_4d_f64 = torch.rand((2, 4, 5), dtype=torch.float64)


_example_inputs1_f32 = (_A_f32, _b_f32, _s1_f32, _s2_f32, _t1_f32, _t2_f32)
_example_inputs1_without_bias_f32 = (_A_f32, _s1_f32, _s2_f32, _t1_f32, _t2_f32)
_example_inputs2_f32 = (
    _A_4d_f32,
    _b_4d_f32,
    _s1_4d_f32,
    _s2_4d_f32,
    _t1_4d_f32,
    _t2_4d_f32,
)
_example_inputs2_without_bias_f32 = (
    _A_4d_f32,
    _s1_4d_f32,
    _s2_4d_f32,
    _t1_4d_f32,
    _t2_4d_f32,
)

_example_inputs1_f64 = (_A_f64, _b_f64, _s1_f64, _s2_f64, _t1_f64, _t2_f64)
_example_inputs1_without_bias_f64 = (_A_f64, _s1_f64, _s2_f64, _t1_f64, _t2_f64)
_example_inputs2_f64 = (
    _A_4d_f64,
    _b_4d_f64,
    _s1_4d_f64,
    _s2_4d_f64,
    _t1_4d_f64,
    _t2_4d_f64,
)
_example_inputs2_without_bias_f64 = (
    _A_4d_f64,
    _s1_4d_f64,
    _s2_4d_f64,
    _t1_4d_f64,
    _t2_4d_f64,
)

_back_substitute_maxpool_naive_f32 = torch.jit.trace(
    _back_substitute_maxpool_naive, _example_inputs1_f32
)
_back_substitute_maxpool_without_bias_naive_f32 = torch.jit.trace(
    _back_substitute_maxpool_without_bias_naive, _example_inputs1_without_bias_f32
)


_back_substitute_maxpool_naive_f64 = torch.jit.trace(
    _back_substitute_maxpool_naive, _example_inputs1_f64
)
_back_substitute_maxpool_without_bias_naive_f64 = torch.jit.trace(
    _back_substitute_maxpool_without_bias_naive, _example_inputs1_without_bias_f64
)


def back_substitute_maxpool2d(
    A: Tensor,
    b: Tensor,
    s1: Tensor,
    s2: Tensor,
    t1: Tensor,
    t2: Tensor,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple,
    dilation: tuple,
    padding: tuple,
    stride: tuple,
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a maxpool operation.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param s1: The slopes of the upper/lower linear relaxation of the maxpool
        operation.
    :param s2: The slopes of the lower/upper linear relaxation of the maxpool
        operation.
    :param t1: The intercepts of the upper/lower linear relaxation of the maxpool
        operation.
    :param t2: The intercepts of the lower/upper linear relaxation of the maxpool
        operation.
    :param input_size: The size of the input tensor.
    :param output_size: The size of the output tensor.
    :param kernel_size: The size of the kernel.
    :param stride: The stride of the maxpool operation.
    :param padding: The padding of the maxpool operation.
    :param dilation: The dilation of the maxpool operation.
    """
    c, hi, wi = input_size
    k = output_size[1] * output_size[2]
    s = kernel_size[0] * kernel_size[1]
    A = A.reshape(-1, c, k)

    if s1.dim() == s2.dim() == 2 and t1.dim() == t2.dim() == 1:
        s1, s2 = s1.reshape(c, k, s), s2.reshape(c, k, s)
        t1, t2 = t1.reshape(c, k), t2.reshape(c, k)

        A, b = (
            (
                _back_substitute_maxpool_naive_f32(A, b, s1, s2, t1, t2)
                if A.dtype == torch.float32
                else _back_substitute_maxpool_naive_f64(A, b, s1, s2, t1, t2)
            )
            if b is not None
            else (
                _back_substitute_maxpool_without_bias_naive_f32(A, s1, s2, t1, t2)
                if A.dtype == torch.float32
                else _back_substitute_maxpool_without_bias_naive_f64(A, s1, s2, t1, t2)
            )
        )

    else:
        raise ValueError(
            f"The dimensions are not supported. "
            f"A: {A.shape}, b: {b.shape}, "
            f"s1: {s1.shape}, s2: {s2.shape}, "
            f"t1: {t1.shape}, t2: {t2.shape}."
        )

    A = F.fold(
        A.reshape(-1, c, k, s).permute(0, 1, 3, 2).reshape(-1, c * s, k),
        input_size[1:],
        kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    ).reshape(-1, c * hi * wi)
    b = b.flatten() if b is not None else None

    return A, b
