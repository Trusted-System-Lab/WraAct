__docformat__ = ["restructuredtext"]
__all__ = ["cal_relaxation_sigmoid", "cal_relaxation_tanh"]

from typing import Callable

import torch
from torch import Tensor

from src.utils import (
    ActRelaxationMode,
    DEEPPOLY,
    CROWN,
    ROVER_SN,
    sigmoid,
    dsigmoid,
    tanh,
    dtanh,
)


def _cal_relaxation_sshape_deeppoly(
    l: Tensor,
    u: Tensor,
    yl: Tensor,
    yu: Tensor,
    kl: Tensor,
    ku: Tensor,
    klu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mask0: Tensor = u == l
    mask1: Tensor = (u <= 0) & ~mask0
    mask2: Tensor = (l >= 0) & ~mask0
    mask3: Tensor = ~(mask0 | mask1 | mask2)

    sl = torch.where(mask0, kl, torch.zeros_like(kl))
    su = sl.clone()

    sl = torch.where(mask1, kl, sl)
    su = torch.where(mask1, klu, su)

    sl = torch.where(mask2, klu, sl)
    su = torch.where(mask2, ku, su)

    sl = torch.where(mask3, torch.minimum(kl, ku), sl)
    su = torch.where(mask3, sl, su)

    tl = -sl * l + yl
    tu = -su * u + yu

    return sl, su, tl, tu


def _cal_relaxation_sshape(
    l: Tensor,
    u: Tensor,
    yl: Tensor,
    yu: Tensor,
    kl: Tensor,
    ku: Tensor,
    klu: Tensor,
    m: Tensor,
    ym: Tensor,
    km: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mask0: Tensor = u == l
    mask1: Tensor = (ku >= klu) & ~mask0
    mask2: Tensor = (kl >= klu) & ~mask0
    mask3: Tensor = ~(mask0 | mask1 | mask2)

    sl = torch.where(mask0, kl, 0.0)
    su = sl.clone()
    tl = torch.where(mask0, yl - sl * l, 0.0)
    tu = tl.clone()

    sl = torch.where(mask1, km, sl)
    su = torch.where(mask1, klu, su)
    tl = torch.where(mask1, ym - sl * m, tl)
    tu = torch.where(mask1, yl - su * l, tu)

    sl = torch.where(mask2, klu, sl)
    su = torch.where(mask2, km, su)
    tl = torch.where(mask2, yu - sl * u, tl)
    tu = torch.where(mask2, ym - su * m, tu)

    sl = torch.where(mask3, kl, sl)
    su = torch.where(mask3, ku, su)
    tl = torch.where(mask3, yl - sl * l, tl)
    tu = torch.where(mask3, yu - su * u, tu)

    return sl, su, tl, tu


def cal_relaxation_sshape(
    l: Tensor,
    u: Tensor,
    f: Callable,
    df: Callable,
    mode: ActRelaxationMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    yl, yu, kl, ku = f(l), f(u), df(l), df(u)

    klu = (yu - yl) / (u - l)

    if mode == DEEPPOLY:
        return _cal_relaxation_sshape_deeppoly(l, u, yl, yu, kl, ku, klu)

    m = (l + u) * 0.5
    ym = f(m)
    km = df(m)

    if mode in {CROWN, ROVER_SN}:
        return _cal_relaxation_sshape(l, u, yl, yu, kl, ku, klu, m, ym, km)

    raise ValueError(f"Unsupported mode: {mode}")


def cal_relaxation_sigmoid(
    l: Tensor,
    u: Tensor,
    mode: ActRelaxationMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return cal_relaxation_sshape(l, u, sigmoid, dsigmoid, mode)


def cal_relaxation_tanh(
    l: Tensor, u: Tensor, mode: ActRelaxationMode
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return cal_relaxation_sshape(l, u, tanh, dtanh, mode)
