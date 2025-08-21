__docformat__ = ["restructuredtext"]
__all__ = ["back_substitute_conv2d"]

import torch.nn.functional as F
from torch import Tensor


def back_substitute_conv2d(
    A: Tensor,
    b: Tensor | None,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    output_padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1,
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a 2D convolution operation.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param weight: The kernel weight of the convolutional operation.
    :param bias: The kernel bias of the convolutional operation.
    :param stride: The stride of the convolutional operation.
    :param padding: The padding of the convolutional operation.
    :param output_padding: The output padding of the convolutional operation.
    :param dilation: The dilation of the convolutional operation.
    :param groups: The groups of the convolutional operation.

    :return: The matrix and bias of the linear relaxation after back-substitution.
    """
    d = (1, 2, 3)
    if b is not None and bias is not None:
        b = b + (A * bias.reshape(1, -1, 1, 1)).sum(dim=d)
    elif bias is not None:
        b = (A * bias.reshape(1, -1, 1, 1)).sum(dim=d)

    A = F.conv_transpose2d(
        A,
        weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    return A, b
