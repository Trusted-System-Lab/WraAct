__docformat__ = ["restructuredtext"]
__all__ = ["cal_relaxation_elu"]

import torch
from torch import Tensor

from ....utils import ActRelaxationMode, elu, delu


def _cal_relaxation_elu(l: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    f, df = elu, delu
    yl, yu = f(l), f(u)

    su = torch.where(l >= 0, 1.0, 0.0)
    sl = su.clone()

    mask = (l <= 0) & (l != u)
    su = torch.where(mask, (yu - yl) / (u - l), su)
    tu = torch.where(mask, -l * su + yl, 0)
    m = (l + u) * 0.5
    sl = torch.where(mask, df(m), sl)
    tl = torch.where(mask, -m * sl + f(m), 0)

    mask = l == u
    sl = torch.where(mask, df(l), sl)
    tl = torch.where(mask, -l * sl + yl, tl)
    su = torch.where(mask, sl, su)
    tu = torch.where(mask, tl, tu)

    return sl, su, tl, tu


def cal_relaxation_elu(
    l: Tensor, u: Tensor, mode: ActRelaxationMode
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _cal_relaxation_elu(l, u)
