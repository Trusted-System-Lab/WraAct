__docformat__ = ["restructuredtext"]
__all__ = ["cal_relaxation_relu"]

import torch
from torch import Tensor

from ....utils import ActRelaxationMode


def _cal_relaxation_relu(l: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Upper bound
    # y <= x
    su = torch.where(l >= 0, 1.0, 0.0)

    # y <= u / (u - l) (x - l)
    mask_u = (l < 0) & (u > 0)
    su = torch.where(mask_u, u / (u - l), su)
    tu = torch.where(mask_u, -l * su, 0)

    # Lower bound
    # y >= 0 or y >= x
    sl = torch.where(u >= -l, 1.0, 0.0)
    tl = torch.zeros_like(l)

    return sl, su, tl, tu


def cal_relaxation_relu(
    l: Tensor,
    u: Tensor,
    mode: ActRelaxationMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _cal_relaxation_relu(l, u)
