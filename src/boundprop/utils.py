__docformat__ = ["restructuredtext"]
__all__ = [
    "cal_conv_data_size",
    "generate_avgpool_weight",
    "unfold_pre_bound_maxpool2d",
    "cal_l_argmax_maxpool2d",
    "get_pool_idxs_maxpool2d",
]

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def cal_conv_data_size(
    input_size: tuple[int, int, int],
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    ceil_mode: bool,
) -> tuple[int, int, tuple]:
    """
    Calculate the output size of the convolutional layer by the input size, kernel
    size, stride, padding, and dilation.
    The calculation is based on the formula from PyTorch as follows:
        - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        - https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    Also, the output padding is calculated based on the formula from PyTorch as
    follows:
        - https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        - https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    :return: The width and height of the output data.
    """
    input_data_size = torch.tensor(input_size[1:])
    kernel_size = torch.tensor(kernel_size)
    stride = torch.tensor(stride)
    padding = torch.tensor(padding)
    dilation = torch.tensor(dilation)

    # Calculate the output size
    temp1 = 2 * padding
    temp2 = dilation * (kernel_size - 1)
    size_before = (input_data_size + temp1 - temp2 - 1) / stride + 1
    output_data_size = (
        torch.ceil(size_before) if ceil_mode else torch.floor(size_before)
    )
    height, width = output_data_size.int().tolist()

    # Calculate the output padding
    output_padding = input_data_size - (
        (output_data_size - 1) * stride - temp1 + temp2 + 1
    )
    output_padding = tuple(output_padding.int().tolist())

    return height, width, output_padding


def generate_avgpool_weight(
    input_channels: int,
    output_channels: int,
    kernel_size: tuple,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Generate the weight for an average pool layer, which is a special case of
    convolution with a kernel of 1/num_inputs. The bias is unnecessary because it is 0.

    :param input_channels: The number of input channels.
    :param output_channels: The number of output channels.
    :param kernel_size: The size of the kernel.
    :param dtype: The data type of the weight.
    :param device: The device of the weight.

    :return: The weight of the average pool layer.
    """
    return torch.empty(
        (input_channels, output_channels, *kernel_size),
        dtype=dtype,
        device=device,
    ).fill_(1.0 / math.prod(kernel_size))


def unfold_pre_bound_maxpool2d(
    l: Tensor,
    u: Tensor,
    nk: int,
    nks: int,
    kernel_size: tuple,
    dilation: tuple,
    padding: tuple,
    stride: tuple,
) -> tuple[Tensor, Tensor]:
    kwargs = {
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
    }
    c = l.shape[0]
    l: Tensor = F.unfold(l, **kwargs)
    u: Tensor = F.unfold(u, **kwargs)
    l = l.reshape((c, nks, nk)).permute(0, 2, 1).reshape((c * nk, nks))
    u = u.reshape((c, nks, nk)).permute(0, 2, 1).reshape((c * nk, nks))

    return l, u


def cal_l_argmax_maxpool2d(l: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """
    Calculate the maximum value and the index of the maximum value of the input tensor.

    :param l: The lower bound of the input tensor.
    :param u: The upper bound of the input tensor.

    :return: The maximum value and the index of the maximum value.
    """
    r = u - l
    l_max, _ = l.max(dim=1)
    max_mask = l == l_max.unsqueeze(1)
    r_masked = r.where(max_mask, torch.tensor(-float("inf")))
    l_argmax = r_masked.argmax(dim=1)

    return l_max, l_argmax


def get_pool_idxs_maxpool2d(
    c: int,
    h: int,
    w: int,
    nks: int,
    nk: int,
    kernel_size: tuple,
    dilation: tuple,
    padding: tuple,
    stride: tuple,
) -> Tensor:
    return (
        F.unfold(
            torch.arange(c * h * w).reshape(c, h, w).to(dtype=torch.float32),
            kernel_size,
            dilation,
            padding,
            stride,
        )
        .reshape((c, nks, nk))
        .permute((0, 2, 1))
        .reshape((c * nk, nks))
        .to(dtype=torch.long)
    )
