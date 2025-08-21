__docformat__ = ["restructuredtext"]
__all__ = ["cal_relaxation_maxpool2d"]

import torch
from torch import Tensor

from ....utils import ActRelaxationMode


def _cal_relaxation_maxpool2d(
    l: Tensor, u: Tensor, l_max: Tensor, l_argmax: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype, device = l.dtype, l.device
    n, s = l.shape

    sl = torch.zeros((n, s), dtype=dtype, device=device)
    tl = torch.zeros((n,), dtype=dtype, device=device)
    su = sl.clone()
    tu = tl.clone()
    idx = torch.arange(n, device=device)

    # Non-zero case. All the lower bounds and upper bounds are zero.
    mask_z = (l == 0).all(dim=1) & (u == 0).all(dim=1)

    # Lower relaxation
    # Lower bound is y > x_i, where x_i has the maximum lower bound.
    mask_l = ~mask_z
    sl[idx[mask_l], l_argmax[mask_l]] = 1.0

    # Upper relaxation
    # (1) Trivial case. one lower bound is larger than the other upper bounds.
    # Set the maximum upper bound to negative infinite to exclude it from the
    # max operation.
    mask_t = ~(mask | mask_z)
    su[idx[mask_t], l_argmax[mask_t]] = 1.0

    # (2) Nontrivial cases.
    # Reference: CROWN and DeepPoly
    # y <= u_max
    tu[mask] = u.max(dim=1)[0][mask]

    return sl, su, tl, tu


def cal_relaxation_maxpool2d(
    l: Tensor,
    u: Tensor,
    mode: ActRelaxationMode,
    l_max: Tensor,
    l_argmax: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n, s = l.shape
    l_max = l_max.flatten()
    l_argmax = l_argmax.flatten()
    mask = mask.flatten()

    dtype = l.dtype
    if dtype == torch.float32:
        sl, su, tl, tu = _cal_relaxation_maxpool2d(l, u, l_max, l_argmax, mask)
    elif dtype == torch.float64:
        sl, su, tl, tu = _cal_relaxation_maxpool2d(l, u, l_max, l_argmax, mask)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    sl = sl.reshape((n, s))
    su = su.reshape((n, s))
    tl = tl.flatten()
    tu = tu.flatten()

    return sl, su, tl, tu
