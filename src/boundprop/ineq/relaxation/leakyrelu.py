__docformat__ = ["restructuredtext"]
__all__ = ["cal_relaxation_leakyrelu"]

import torch
from torch import Tensor

from ....utils import ActRelaxationMode, leakyrelu, dleakyrelu


def _cal_relaxation_leakyrelu(l: Tensor, u: Tensor, alpha: float):
    f, df = leakyrelu, dleakyrelu
    yl = f(l)
    # Upper bound
    su = torch.where(l >= 0, 1.0, 0.0)
    su = torch.where(u <= 0, alpha, su)

    # y <= u / (u - l) (x - l)
    mask_u = (l < 0) & (u > 0)
    su = torch.where(mask_u, (u - yl) / (u - l), su)
    tu = torch.where(mask_u, -l * su + yl, 0)

    # Lower bound
    # y >= 0 or y >= x
    sl = torch.where(u > -l, 1.0, alpha)
    tl = torch.zeros_like(l)

    return sl, su, tl, tu


def cal_relaxation_leakyrelu(
    l: Tensor, u: Tensor, mode: ActRelaxationMode, alpha: float = 0.01
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _cal_relaxation_leakyrelu(l, u, alpha)
