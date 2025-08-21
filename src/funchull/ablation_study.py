__docformat__ = "restructuredtext"
__all__ = [
    "SigmoidHullA",
    "SigmoidHullB",
    "TanhHullA",
    "TanhHullB",
    "ELUHullA",
    "MaxPoolDLPHullA",
]

import numpy as np
from numpy import ndarray

from . import SShapeHull, MaxPoolHullDLP
from .acthull import ELUHull
from ..utils import *

_TOL = 1e-4
_LLRELU_ALPHA = 0.01
_MIN_BOUNDS_RANGE = 0.05
_MIN_DLP_ANGLE = 0.1
_ELU_MAX_AUX_POINT = -1.25
_DEBUG = False


class SShapeHullA(SShapeHull):
    @classmethod
    def _construct_dlp_case1(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,
        ndarray,
        None,
        float,
        ndarray | None,
    ]:
        f = cls._f
        blu2, klu2, su = cls._get_parallel_tangent_line(klui, get_big=False)
        su = (su + xli) / 2.0  # For ablation study
        kp1, kli = (yli - f(su)) / (xli - su), kli
        bp1, bli = yli - kp1 * xli, yli - kli * xli

        if xui > 0:
            x = np.asarray([xli, su, xui], dtype=np.float64)
            b, k, _ = cls._get_second_tangent_line(x, get_big=True)
            blui, bp2, bui = b
            klui, kp2, kui = k
        else:
            kp2, klui, kui = (yui - f(su)) / (xui - su), klui, kui
            bp2, blui, bui = yui - kp2 * xui, yli - klui * xli, yui - kui * xui

        aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_l[:, 0] = [bli]
        aux_lines_l[:, idx + 1] = [kli]
        aux_lines_l[:, -1] = -1.0
        aux_point_l = None

        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [yli - klui * xli]
            aux_lines_u[:, idx + 1] = [klui]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = None
        else:
            aux_lines_u = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [bp1, bp2]
            aux_lines_u[:, idx + 1] = [kp1, kp2]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = su

        if return_single_neuron_constrs:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [-1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    @classmethod
    def _construct_dlp_case2(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,
        ndarray,
        float,
        None,
        ndarray | None,
    ]:
        f = cls._f
        blu2, klu2, sl = cls._get_parallel_tangent_line(klui, get_big=True)
        sl = (sl + xui) / 2.0  # For ablation study
        kp1, kui = (yui - f(sl)) / (xui - sl), kui
        bp1, bui = yui - kp1 * xui, yui - kui * xui

        if xli < 0:
            x = np.asarray([xui, sl, xli], dtype=np.float64)
            b, k, _ = cls._get_second_tangent_line(x, get_big=False)
            blui, bp2, bli = b
            klui, kp2, kli = k
        else:
            kp2, klui, kli = (yli - f(sl)) / (xli - sl), klui, kli
            bp2, blui, bli = yli - kp2 * xli, yui - klui * xui, yli - kli * xli

        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [yui - klui * xui]
            aux_lines_l[:, idx + 1] = [klui]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = None
        else:
            aux_lines_l = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bp1, bp2]
            aux_lines_l[:, idx + 1] = [kp1, kp2]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = sl

        aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_u[:, 0] = [bui]
        aux_lines_u[:, idx + 1] = [kui]
        aux_lines_u[:, -1] = -1.0
        aux_point_u = None

        if return_single_neuron_constrs:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [1.0, -1.0, -1.0, -1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    @classmethod
    def _construct_dlp_case3(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,
        ndarray,
        float,
        float,
        ndarray | None,
    ]:
        f = cls._f
        blul, klul, su = cls._get_parallel_tangent_line(klui, get_big=False)
        bluu, kluu, sl = cls._get_parallel_tangent_line(klui, get_big=True)

        su = (su + xli) / 2.0  # For ablation study
        sl = (sl + xui) / 2.0  # For ablation study

        x_temp = np.asarray([xli, su], dtype=np.float64)
        b_temp, k_temp, _ = cls._get_second_tangent_line(x_temp, get_big=True)
        btu, bp2u = b_temp
        ktu, kp2u = k_temp
        x_temp = np.asarray([xui, sl], dtype=np.float64)
        b_temp, k_temp, _ = cls._get_second_tangent_line(x_temp, get_big=False)
        btl, bp2l = b_temp
        ktl, kp2l = k_temp

        kp1u, kp1l = (yli - f(su)) / (xli - su), (yui - f(sl)) / (xui - sl)
        bp1u, bp1l = yli - kp1u * xli, yui - kp1l * xui

        if abs((kp1u - kp2u) / (1 - kp1u * kp2u)) < _MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [yli - klui * xli]
            aux_lines_u[:, idx + 1] = [klui]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = None
        else:
            aux_lines_u = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [bp1u, bp2u]
            aux_lines_u[:, idx + 1] = [kp1u, kp2u]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = su

        if abs((kp1l - kp2l) / (1 - kp1l * kp2l)) < _MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [yui - klui * xui]
            aux_lines_l[:, idx + 1] = [klui]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = None
        else:
            aux_lines_l = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bp1l, bp2l]
            aux_lines_l[:, idx + 1] = [kp1l, kp2l]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = sl

        if return_single_neuron_constrs:
            bli, bui = yli - kli * xli, yui - kui * xui
            temp = np.zeros((6, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [bui, btu, bluu, bli, btl, blul]
            temp[:, idx + 1] = [kui, ktu, kluu, kli, ktl, klul]
            temp[:, -1] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c


class SShapeHullB(SShapeHull):
    @classmethod
    def _construct_dlp_case1(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,
        ndarray,
        None,
        float,
        ndarray | None,
    ]:

        f = cls._f
        blu2, klu2, su = cls._get_parallel_tangent_line(klui, get_big=False)
        su = su / 2.0  # For ablation study

        kp1, kli = (yli - f(su)) / (xli - su), kli
        bp1, bli = yli - kp1 * xli, yli - kli * xli

        if xui > 0:
            x = np.asarray([xli, su, xui], dtype=np.float64)
            b, k, _ = cls._get_second_tangent_line(x, get_big=True)
            blui, bp2, bui = b
            klui, kp2, kui = k
        else:
            kp2, klui, kui = (yui - f(su)) / (xui - su), klui, kui
            bp2, blui, bui = yui - kp2 * xui, yli - klui * xli, yui - kui * xui

        aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_l[:, 0] = [bli]
        aux_lines_l[:, idx + 1] = [kli]
        aux_lines_l[:, -1] = -1.0
        aux_point_l = None

        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [yli - klui * xli]
            aux_lines_u[:, idx + 1] = [klui]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = None
        else:
            aux_lines_u = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [bp1, bp2]
            aux_lines_u[:, idx + 1] = [kp1, kp2]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = su

        if return_single_neuron_constrs:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [-1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    @classmethod
    def _construct_dlp_case2(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,
        ndarray,
        float,
        None,
        ndarray | None,
    ]:
        f = cls._f
        blu2, klu2, sl = cls._get_parallel_tangent_line(klui, get_big=True)
        sl = sl / 2.0  # For ablation study

        kp1, kui = (yui - f(sl)) / (xui - sl), kui
        bp1, bui = yui - kp1 * xui, yui - kui * xui

        if xli < 0:
            x = np.asarray([xui, sl, xli], dtype=np.float64)
            b, k, _ = cls._get_second_tangent_line(x, get_big=False)
            blui, bp2, bli = b
            klui, kp2, kli = k
        else:
            kp2, klui, kli = (yli - f(sl)) / (xli - sl), klui, kli
            bp2, blui, bli = yli - kp2 * xli, yui - klui * xui, yli - kli * xli

        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [yui - klui * xui]
            aux_lines_l[:, idx + 1] = [klui]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = None
        else:
            aux_lines_l = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bp1, bp2]
            aux_lines_l[:, idx + 1] = [kp1, kp2]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = sl

        aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_u[:, 0] = [bui]
        aux_lines_u[:, idx + 1] = [kui]
        aux_lines_u[:, -1] = -1.0
        aux_point_u = None

        if return_single_neuron_constrs:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [1.0, -1.0, -1.0, -1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    @classmethod
    def _construct_dlp_case3(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,
        ndarray,
        float,
        float,
        ndarray | None,
    ]:

        f = cls._f
        blul, klul, su = cls._get_parallel_tangent_line(klui, get_big=False)
        bluu, kluu, sl = cls._get_parallel_tangent_line(klui, get_big=True)
        su = su / 2.0  # For ablation study
        sl = sl / 2.0  # For ablation study

        x_temp = np.asarray([xli, su], dtype=np.float64)
        b_temp, k_temp, _ = cls._get_second_tangent_line(x_temp, get_big=True)
        btu, bp2u = b_temp
        ktu, kp2u = k_temp
        x_temp = np.asarray([xui, sl], dtype=np.float64)
        b_temp, k_temp, _ = cls._get_second_tangent_line(x_temp, get_big=False)
        btl, bp2l = b_temp
        ktl, kp2l = k_temp

        kp1u, kp1l = (yli - f(su)) / (xli - su), (yui - f(sl)) / (xui - sl)
        bp1u, bp1l = yli - kp1u * xli, yui - kp1l * xui

        if abs((kp1u - kp2u) / (1 - kp1u * kp2u)) < _MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [yli - klui * xli]
            aux_lines_u[:, idx + 1] = [klui]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = None
        else:
            aux_lines_u = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [bp1u, bp2u]
            aux_lines_u[:, idx + 1] = [kp1u, kp2u]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = su

        if abs((kp1l - kp2l) / (1 - kp1l * kp2l)) < _MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [yui - klui * xui]
            aux_lines_l[:, idx + 1] = [klui]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = None
        else:
            aux_lines_l = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bp1l, bp2l]
            aux_lines_l[:, idx + 1] = [kp1l, kp2l]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = sl

        if return_single_neuron_constrs:
            bli, bui = yli - kli * xli, yui - kui * xui
            temp = np.zeros((6, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [bui, btu, bluu, bli, btl, blul]
            temp[:, idx + 1] = [kui, ktu, kluu, kli, ktl, klul]
            temp[:, -1] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c


class SigmoidHullA(SShapeHullA):

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_second_tangent_line(x1, get_big, "sigmoid")

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_parallel_tangent_line(k, get_big, "sigmoid")

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return sigmoid(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dsigmoid(x)


class SigmoidHullB(SShapeHullB):

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_second_tangent_line(x1, get_big, "sigmoid")

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_parallel_tangent_line(k, get_big, "sigmoid")

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return sigmoid(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dsigmoid(x)


class TanhHullA(SShapeHullA):

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_second_tangent_line(x1, get_big, "tanh")

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_parallel_tangent_line(k, get_big, "tanh")

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return tanh(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dtanh(x)


class TanhHullB(SShapeHullB):

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_second_tangent_line(x1, get_big, "tanh")

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_parallel_tangent_line(k, get_big, "tanh")

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return tanh(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dtanh(x)


class ELUHullA(ELUHull):
    @classmethod
    def _construct_dlp(
        cls, idx: int, dim: int, l: float, u: float
    ) -> tuple[ndarray, float | None]:

        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)

        yl = cls._f(l)
        yu = cls._f(u)
        if l >= _ELU_MAX_AUX_POINT or u <= _ELU_MAX_AUX_POINT:
            k = (yu - yl) / (u - l)
            b = yu - k * u
            aux_lines = np.asarray(
                [[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64
            )
            return aux_lines, None

        # The intersection point of the two linear pieces should not be positive to
        # avoid large coefficients in the upper bound because the linear pieces are
        # too close to the upper bound.
        m = (l + u) / 2.0
        m = (m + l) / 2.0  # For ablation study
        m = min(m, _ELU_MAX_AUX_POINT)

        kp1 = (yu - cls._df(m)) / (u - m)
        bp1 = yu - kp1 * u
        kp2 = (yl - cls._df(m)) / (l - m)
        bp2 = yl - kp2 * l

        # Estimate the angle of the two linear pieces to avoid large coefficients.
        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            k = (yu - yl) / (u - l)
            b = yu - k * u
            aux_lines = np.asarray(
                [[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64
            )
            return aux_lines, None

        aux_lines = np.asarray(
            [
                [bp1] + temp1 + [kp1] + temp2 + [-1.0],
                [bp2] + temp1 + [kp2] + temp2 + [-1.0],
            ],
            dtype=np.float64,
        )
        aux_point = m

        return aux_lines, aux_point


class MaxPoolDLPHullA(MaxPoolHullDLP):
    @classmethod
    def _construct_dlp(cls, v: ndarray, nt_idxs: list[int]) -> ndarray:
        # When constructing the DLP function as the upper bound of the MaxPool
        # function, we only need to consider the nontrivial coordinates.

        d = v.shape[1] - 1

        # Get the lower and upper bounds of the non-trivial indices based on vertices.
        l = np.min(v[:, 1:], axis=0)
        u = np.max(v[:, 1:], axis=0)

        r = u - l
        # Get the indices of r in descending order.
        ordered_r_idx = np.argsort(r)[::-1]

        # Remove the trivial cases.
        ordered_r_idx = np.asarray([idx for idx in ordered_r_idx if idx in nt_idxs])

        # Group ordered_r_idx into two groups:
        # the first group contains the odd indices of ordered_r_idx,
        # the second group contains the even indices of ordered_r_idx.
        n = len(ordered_r_idx)
        r_idx1 = ordered_r_idx[: n // 2]  # For ablation study
        r_idx2 = ordered_r_idx[n // 2 :]  # For ablation study

        dlp_lines = np.zeros((2, 2 + d), dtype=np.float64)
        # y >= \sum x_{r_idx_odd}
        # y >= \sum x_{r_idx_even}
        dlp_lines[:, -1] = 1.0
        dlp_lines[0, r_idx1 + 1] = -1.0
        dlp_lines[1, r_idx2 + 1] = -1.0

        return dlp_lines
