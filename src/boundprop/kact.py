"""
This module provides the functions to generate the groups of neurons and calculate the
convex hulls of the grouped neurons.

.. seealso::

    This neuron grouping strategy framework is proposed by

    - `Beyond the single neuron convex barrier for neural network certification
      <https://proceedings.neurips.cc/paper_files/paper/2019/file
      /0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf>`__

    - `PRIMA: General and Precise Neural Network Certification via Scalable
      Convex Hull Approximations <https://dl.acm.org/doi/pdf/10.1145/3498704>`__
"""

__docformat__ = "restructuredtext"
__all__ = [
    "generate_groups_lp",
    "back_substitute_grouped_constrs",
    "cal_grouped_acthull",
    "back_substitute_to_input_kact",
]

import itertools
import logging
import multiprocessing
import time
from typing import Any

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from .ineq.backsub import *
from ..funchull import *
from ..utils import *


class _Cache:
    # This contains the covers of the neurons with a partition size.
    # The key is the partition (partition_size, group_dim, max_overlap_size).
    # The value is the list of the groups of neuron indices.
    sparse_covers: dict[tuple[int, int, int], list[list[int]]] = {}

    # We store this cache in disk, because they are universal.
    # This contains the templates of the constraints of the input variables with a group
    # size of k.
    # The key is the group size.
    # The value is the template tensor.
    hexagon_templates: dict[int, Tensor] = {}
    rectangle_templates: dict[int, Tensor] = {}
    rhombus_templates: dict[int, Tensor] = {}


def generate_groups_lp(
    pre_l: Tensor,
    pre_u: Tensor,
    mask_mn: Tensor,
    act_type: ActivationType,
    kact_args: KActLPArgs,
    pool_input_ids: Tensor | None = None,
) -> Tensor | None:
    """
    This function generates the groups of neurons for the k-activation function.

    :param pre_l: The lower bounds of the pre-activation values.
    :param pre_u: The upper bounds of the pre-activation values.
    :param mask_mn: The mask of the neurons that are not trivial for multi-neuron
        constraints.
    :param act_type: The type of the activation function. Refer to
        :class:`ActivationType` for more details.
    :param kact_args: The arguments of the k-activation function. Refer to
        :class:`KActLPArgs` for more details.
    :param pool_input_ids: The indices of the neurons in the max pooling layer.

    :return: The grouped indices of the neurons.
    """
    logger = logging.getLogger("rover")
    logger.debug(f"Start generating groups for {act_type}.")
    time_start = time.perf_counter()

    # Filter out unnecessary neurons
    if act_type in {
        ActivationType.RELU,
        ActivationType.LEAKY_RELU,
        ActivationType.ELU,
        ActivationType.SIGMOID,
        ActivationType.TANH,
    }:
        # Filter out the neurons with big ranges.
        ids = torch.arange(mask_mn.numel(), device=mask_mn.device)[mask_mn]

        r = (pre_u - pre_l)[mask_mn]
        logger.debug(
            f"Max range {torch.max(r).item():.2f}, Min range {torch.min(r).item():.2f}"
        )
        ids = ids[r.argsort(descending=True)]
        grouped_ids = _generate_grouped_ids(
            ids,
            kact_args.partition_size,
            kact_args.group_size,
            kact_args.max_overlap_size,
        )

    elif act_type == ActivationType.MAXPOOL2D:
        grouped_ids = pool_input_ids[mask_mn]

    else:
        raise NotImplementedError(f"Activation function {act_type} is not supported.")

    grouped_ids = grouped_ids[: kact_args.max_groups].to(pre_l.device)

    logger.debug(
        f"Finish generating {grouped_ids.size(0)} groups for {act_type}. "
        f"Cost time: {time.perf_counter() - time_start:.4f}s"
    )

    return grouped_ids


def _generate_grouped_ids(
    ids: Tensor, partition_size: int, group_dim: int, max_overlap_size: int
) -> Tensor:
    n = ids.size(0)
    n_partition = n // partition_size

    grouped_ids = torch.empty((0, group_dim), dtype=torch.long, device=ids.device)
    if n_partition != 0:
        grouped_ids = torch.vstack(
            [
                grouped_ids,
                _generate_grouped_ids_in_a_partition(
                    ids[: n_partition * partition_size].reshape(
                        n_partition, partition_size
                    ),
                    group_dim,
                    max_overlap_size,
                ),
            ]
        )

    if n - n_partition * partition_size >= group_dim:
        # The remaining neurons are enough to form a group
        grouped_ids = torch.vstack(
            [
                grouped_ids,
                _generate_grouped_ids_in_a_partition(
                    ids[n_partition * partition_size :].reshape(1, -1),
                    group_dim,
                    max_overlap_size,
                ),
            ]
        )

    return grouped_ids


def _generate_grouped_ids_in_a_partition(
    ids: Tensor, group_dim: int, max_overlap_size: int
) -> Tensor:
    id_groups = torch.tensor(
        _get_sparse_groups(ids.size(1), group_dim, max_overlap_size),
        dtype=torch.long,
        device=ids.device,
    )
    a = ids.unsqueeze(1).expand(-1, id_groups.size(0), -1)
    b = id_groups.unsqueeze(0).expand(ids.size(0), -1, -1)
    grouped_ids_ = a.gather(2, b)
    return grouped_ids_.reshape(-1, group_dim)


def _get_sparse_groups(
    partition_size: int, group_dim: int, max_overlap_size: int
) -> list[list[int]]:
    if (partition_size, group_dim, max_overlap_size) in _Cache.sparse_covers:
        return _Cache.sparse_covers[(partition_size, group_dim, max_overlap_size)]

    id_groups = []
    for new_group in itertools.combinations(list(range(partition_size)), group_dim):
        if all(
            len(set(new_group).intersection(set(existed_group))) <= max_overlap_size
            for existed_group in id_groups
        ):
            id_groups.append(list(new_group))

    _Cache.sparse_covers[(partition_size, group_dim, max_overlap_size)] = id_groups
    return id_groups


def _get_constrs_template(
    group_dim: int, shape: ConstrTemplate = ConstrTemplate.HEXAGON
) -> Tensor:
    """
    This function provides a hard-coded template for the constraints of the input
    variables.
    For an example of hexagon with a group_dim=3, the template is:

    .. code-block:: python
        tensor(
            [
                [ 1,  1,  1], [ 1,  1,  0], [ 1,  1, -1], [ 1,  0,  1],
                [ 1,  0, -1], [ 1, -1,  1], [ 1, -1,  0], [ 1, -1, -1],
                [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1],
                [-1,  1,  1], [-1,  1,  0], [-1,  1, -1], [-1,  0,  1],
                [-1,  0, -1], [-1, -1,  1], [-1, -1,  0], [-1, -1, -1]
            ]
        )

    .. attention::
        The output template should have symmetrical when ordering the constraints.
        Because the other functions will use such symmetry to reduce the computation.
    """
    if shape == ConstrTemplate.HEXAGON and group_dim in _Cache.hexagon_templates:
        return _Cache.hexagon_templates[group_dim]

    elif shape == ConstrTemplate.RHOMBUS and group_dim in _Cache.rhombus_templates:
        return _Cache.rhombus_templates[group_dim]

    # NOTE：KEEP the symmetry of the template.
    coeffs = torch.tensor([1, 0, -1])
    template = torch.cartesian_prod(*([coeffs] * group_dim))

    if shape == ConstrTemplate.HEXAGON:
        mask = template.abs().sum(dim=1) > 1
        template = template[mask]
        _Cache.hexagon_templates[group_dim] = template
    elif shape == ConstrTemplate.RHOMBUS:
        mask = template.abs().sum(dim=1) == group_dim
        template = template[mask]
        _Cache.rhombus_templates[group_dim] = template
    else:
        raise NotImplementedError(f"InputConstrsTemplate {shape} is not supported.")

    return template


def back_substitute_grouped_constrs(
    input_bound: ScalarBound,
    pre_module: Any,
    n_vars: int,
    ids_grouped: Tensor,
    kact_max_parallel_groups: int,
    constr_template_shape: ConstrTemplate,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    This function calculates the input constraints of the grouped neurons by backward
    inequality propagation.
    The input constraints are the constraints of the input variables that are related to
    the grouped neurons.
    The input constraints are calculated by the backward inequality propagation.

    :param input_bound: The bound of the input variables.
    :param pre_module: The previous linear module and it should be
        :class:`LinearIneqNode` or :class:`LinearIneqConv2d`.
    :param n_vars: The number of input variables.
    :param ids_grouped: The grouped indices of the neurons.
    :param kact_max_parallel_groups: The maximum number of groups to be processed in
        parallel.
    :param constr_template_shape: The template of the input constraints. Refer to
        :class:`ConstrTemplate` for more details.
    :param dtype: The data type of the input constraints.
    :param device: The device of the input constraints.

    :return: The input constraints of the grouped neurons.
    """
    logger = logging.getLogger("rover")
    logger.debug("Start calculating input constraints of convex hulls.")
    time_start = time.time()

    # Calculate the input constrs from template by backward inequality propagation.
    # Here we control the maximum number of groups to be processed in parallel.
    k = ids_grouped.size(1)
    template = _get_constrs_template(k, shape=constr_template_shape).to(
        dtype=dtype, device=device
    )
    ori_template = template.clone()

    # Because the template is symmetrical, we only need half of it.
    # Do not use negative template here, because it may cause the float error.
    template = template[: template.size(0) // 2]
    n_t = template.size(0)  # The number of template constraints

    # Allocate the constrs template for each group.
    # The input constraints has three dimension of (g, n_t, k+1), where g is the number
    # of groups.
    max_n_g = kact_max_parallel_groups
    # The variable to store the input constraints of the grouped neurons.
    constrs = torch.empty((0, 2 * n_t, k + 1), dtype=dtype, device=device)
    n_parallel = (ids_grouped.size(0) - 1) // max_n_g + 1
    for i in range(n_parallel):
        start, end = i * max_n_g, (i + 1) * max_n_g
        ids_grouped_partial = ids_grouped[start:end]
        n_g = ids_grouped_partial.size(0)

        # Fill template in the input constraints with extended dimension for bound
        # propagation.
        ineqs = torch.zeros((n_g, n_t, n_vars), dtype=dtype, device=device)
        idxs_g = torch.arange(n_g, device=device).reshape(n_g, 1, 1)
        idxs_t = torch.arange(n_t, device=device).reshape(1, n_t, 1)
        ids_gp = ids_grouped_partial.reshape(n_g, 1, k)
        ineqs[idxs_g, idxs_t, ids_gp] = template

        # Calculate the biases of constrs by backward inequality propagation.
        ineqs = ineqs.reshape((-1, n_vars))
        constr = LConstr(A=ineqs)
        bound = back_substitute_to_input_kact(
            pre_module, LConstrBound(L=constr, U=constr), input_bound
        )
        l, u = bound.l, bound.u
        # if l.max() < -1e6 or u.max() > 1e6:
        #     raise RuntimeError("The bounds are too large and maybe a bug.")
        # Only retain the coefficients of the input variables in the ids_grouped.
        # Format: b + Ax >= 0
        l = -l.reshape((n_g, n_t, 1))
        u = torch.flip(u.reshape((n_g, n_t, 1)), dims=[1])
        const = torch.cat((l, u), dim=1)
        coeffs = (
            ori_template.clone()
            .reshape((2 * n_t, k))
            .repeat((n_g, 1))
            .reshape((n_g, 2 * n_t, k))
        )

        new_constrs = torch.cat((const, coeffs), dim=2)
        constrs = torch.cat((constrs, new_constrs), dim=0)

    logger.debug(
        f"Finish calculating input constraints of convex hulls in "
        f"{time.time() - time_start:.4f}s."
    )

    return constrs


def cal_grouped_acthull(
    pre_l: Tensor,
    pre_u: Tensor,
    mask_mn: Tensor,
    grouped_ids: Tensor,
    grouped_input_constrs: Tensor,
    act_type: ActivationType,
    pool_input_l: Tensor | None = None,
    pool_input_u: Tensor | None = None,
    use_multi_threads: bool = True,
    return_trivial_pool_idxs: bool = False,
) -> list[ndarray] | tuple[list[ndarray], list[int] | None]:
    """
    This function calculates the function hulls of the grouped neurons.

    :param return_trivial_pool_idxs:
    :param pre_l: The lower bounds of the pre-activation values.
    :param pre_u: The upper bounds of the pre-activation values.
    :param mask_mn: The mask of the neurons that are not trivial for multi-neuron
        constraints.
    :param grouped_ids: The grouped indices of the neurons.
    :param grouped_input_constrs: The input constraints of the grouped neurons.
    :param act_type: The type of the activation function. Refer to
        :class:`ActivationType` for more details.
        constraint, which is used for bound propagation with multi-neuron constraints.
    :param pool_input_l: The lower bounds of the input neurons in the max pooling layer.
    :param pool_input_u: The upper bounds of the input neurons in the max pooling layer.
    :param use_multi_threads: Whether to use multi-threads to calculate the convex
        hulls.

    .. attention::
        This function has some experimental features that are not implemented yet.

    :return: The function hulls of the grouped neurons.
    """
    logger = logging.getLogger("rover")
    logger.debug("Start calculating convex hulls.")
    time_start = time.perf_counter()

    fun_hull = _get_func_hull(act_type)
    grouped_l, grouped_u = _group_input_bounds(
        act_type, pre_l, pre_u, mask_mn, grouped_ids, pool_input_l, pool_input_u
    )
    results = _cal_func_hull(
        fun_hull, grouped_input_constrs, grouped_l, grouped_u, use_multi_threads
    )

    logger.debug(
        f"Finish calculating convex hulls in "
        f"{time.perf_counter() - time_start:.4f}s."
    )

    grouped_constrs = results

    if return_trivial_pool_idxs:
        trivial_pool_idxs = None
        if act_type == ActivationType.MAXPOOL2D:
            grouped_constrs, trivial_pool_idxs = _collect_trivial_pool_idxs(
                grouped_constrs, mask_mn
            )
        return grouped_constrs, trivial_pool_idxs

    return grouped_constrs


def _get_func_hull(act_type: ActivationType) -> ActHull:
    hull_classes = {
        ActivationType.RELU: ReLUHull,
        ActivationType.LEAKY_RELU: LeakyReLUHull,
        ActivationType.ELU: ELUHull,
        ActivationType.SIGMOID: SigmoidHull,
        ActivationType.TANH: TanhHull,
        ActivationType.MAXPOOL2D: MaxPoolHullDLP,
    }

    if act_type not in hull_classes:
        raise NotImplementedError(f"Activation function {act_type} is not supported.")

    HullClass = hull_classes[act_type]
    return HullClass()


def _group_input_bounds(
    act_type: ActivationType,
    pre_l: Tensor,
    pre_u: Tensor,
    mask_mn: Tensor,
    grouped_ids: Tensor,
    pool_input_l: Tensor | None = None,
    pool_input_u: Tensor | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    # Group lower bounds and upper bounds by the ids_grouped.
    if act_type != ActivationType.MAXPOOL2D:
        grouped_ids = grouped_ids.cpu().numpy()
        pre_l = pre_l.cpu().numpy()
        pre_u = pre_u.cpu().numpy()
        grouped_l = [pre_l[ids] for ids in grouped_ids]
        grouped_u = [pre_u[ids] for ids in grouped_ids]
    else:
        pool_input_l_mn = pool_input_l[mask_mn].cpu().numpy()
        pool_input_u_mn = pool_input_u[mask_mn].cpu().numpy()
        num_pool = pool_input_l_mn.shape[0]
        # The pre_l and pre_u of MaxPool have been grouped by pools.
        grouped_l = [pool_input_l_mn[i] for i in range(num_pool)]
        grouped_u = [pool_input_u_mn[i] for i in range(num_pool)]
    return grouped_l, grouped_u


def _cal_func_hull(
    fun_hull: ActHull,
    grouped_input_constrs: Tensor,
    grouped_l: list[Tensor],
    grouped_u: list[Tensor],
    use_multi_threads: bool,
) -> list[ndarray] | list[tuple[ndarray, ndarray, ndarray]]:
    grouped_input_constrs = grouped_input_constrs.detach().cpu().numpy()
    # The number of output constraints in each group may be different. So we use a
    # list to store the output constraints. But that is for LP. For BP, the number of
    # output constraints is fixed because we want parallel  computing.

    if use_multi_threads:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(
                fun_hull.cal_hull, zip(grouped_input_constrs, grouped_l, grouped_u)
            )
        return list(results)
    i = 0
    results = []
    for input_constrs, l, u in zip(grouped_input_constrs, grouped_l, grouped_u):
        output_constrs = fun_hull.cal_hull(input_constrs, l, u)
        results.append(output_constrs)
        i += 1
    return results


def _collect_trivial_pool_idxs(
    grouped_constrs: list[ndarray],
    mask_mn: Tensor,
) -> tuple[list[ndarray], list[int]]:
    """
    Collect the trivial pool detected by the function hull algorithm.
    """

    logger = logging.getLogger("rover")

    # Find the next non-None element to get the shape of the constraints.
    pool_mask_idxs = torch.arange(len(mask_mn), device=mask_mn.device)[mask_mn]
    trivial_pool_idxs = []
    for constr, pool_idxs in zip(grouped_constrs, pool_mask_idxs):
        if constr is None:
            continue
        mask_nz = ~np.isclose(constr, 0)
        num_nz = mask_nz.sum(axis=1)
        if np.all(num_nz == 2):  # The trivial pool of y = const or y = x_i.
            trivial_pool_idxs.append(pool_idxs.item())

    logger.debug(f"Find {len(trivial_pool_idxs)} trivial pools.")

    return grouped_constrs, trivial_pool_idxs


def back_substitute_to_input_kact(
    self: "BasicIneqNode",  # noqa
    constr_bound: LConstrBound,
    input_bound: ScalarBound,
) -> ScalarBound:
    """
    Back-substitute the linear relaxation to the preceding layer until the input layer.

    :param input_bound:
    :param self: The current layer object, and it is a linear layer.
    :param constr_bound: The linear relaxation to back-substitute.

    :return: The linear relaxation represented by input variables.
    """
    logger = logging.getLogger("rover")
    logger.debug(f"Back-substitute to input for {self}.")
    start = time.perf_counter()

    n = constr_bound.L.A.shape[0]
    dtype, device = constr_bound.L.A.dtype, constr_bound.L.A.device
    bound = ScalarBound(
        l=torch.full((n,), -torch.inf, dtype=dtype, device=device),
        u=torch.full((n,), torch.inf, dtype=dtype, device=device),
    )

    in_residual_block = False
    constr_bound_r: LConstrBound | None = None
    residual_second_path: list["BasicIneqNode"] = []  # noqa
    module: "BasicIneqNode" = self  # noqa
    while True:
        if (
            constr_bound_r is not None
            and module.next_nodes is not None
            and len(module.next_nodes) == 2
        ):
            constr_bound, constr_bound_r, residual_second_path, new_bound = (
                back_substitute_residual_second_path(
                    self,
                    module,
                    constr_bound,
                    constr_bound_r,
                    residual_second_path,
                    store_updated_bounds=False,
                )
            )
            in_residual_block = False

            if new_bound is not None:
                bound = bound.intersect(new_bound)

        constr_bound, new_bound = back_substitute_once_with_update_bound(
            self, module, constr_bound, in_residual_block, store_updated_bounds=False
        )

        if new_bound is not None:
            bound = bound.intersect(new_bound)

        if module.pre_nodes is None:
            break

        if len(module.pre_nodes) == 2:
            constr_bound_r, residual_second_path = collect_residual_second_path(
                module, constr_bound, residual_second_path
            )
            in_residual_block = True
        elif len(module.pre_nodes) != 1:
            raise RuntimeError(f"Invalid number of pre nodes: {len(module.pre_nodes)}.")

        module = module.pre_nodes[0]

    new_bound, _ = self.cal_bounds(constr_bound, input_bound)
    bound = bound.intersect(new_bound)

    logger.debug(
        f"Finish back-substitution in {time.perf_counter() - start:.4f} seconds."
    )

    return bound
