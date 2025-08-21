__docformat__ = ["restructuredtext"]
__all__ = [
    "back_substitute_to_input",
    "back_substitute_once_with_update_bound",
    "collect_residual_second_path",
    "back_substitute_residual_second_path",
]

import logging
import time

from ...base import NonLinearNode
from ....utils import LConstrBound, ScalarBound


def back_substitute_to_input(
    self: "BasicIneqNode",  # noqa
    constr_bound: LConstrBound,
) -> LConstrBound:
    """
    Back-substitute the linear relaxation to the preceding layer until the input layer.

    :param self: The current layer object, and it is a linear layer.
    :param constr_bound: The linear relaxation to back-substitute.

    :return: The linear relaxation represented by input variables.
    """
    logger = logging.getLogger("rover")
    logger.debug(f"Back-substitute to input for {self}.")
    start = time.perf_counter()

    # The constraint bound for the output of the residual block.
    constr_bound_r: LConstrBound | None = None
    # The queue to store the modules in the second path of the residual block.
    residual_second_path: list["BasicIneqNode"] = []  # noqa
    # If self is after a residual block and the current module is in the residual block,
    # then do not update the bound by the current module.
    in_residual_block = False

    module: "BasicIneqNode" = self  # noqa
    while True:
        if (
            constr_bound_r is not None
            and module.next_nodes is not None
            and len(module.next_nodes) == 2
        ):
            constr_bound, constr_bound_r, residual_second_path, _ = (
                back_substitute_residual_second_path(
                    self, module, constr_bound, constr_bound_r, residual_second_path
                )
            )
            in_residual_block = False

        constr_bound, _ = back_substitute_once_with_update_bound(
            self, module, constr_bound, in_residual_block
        )

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

    logger.debug(
        f"Finish back-substitution in {time.perf_counter() - start:.4f} seconds."
    )

    return constr_bound


def back_substitute_once_with_update_bound(
    self: "BasicIneqNode",  # noqa
    module: "BasicIneqNode",  # noqa
    constr_bound: LConstrBound,
    in_residual_block: bool,
    store_updated_bounds: bool = True,
) -> LConstrBound | tuple[LConstrBound, ScalarBound | None]:
    logger = logging.getLogger("rover")
    constr_bound = module.back_substitute_once(constr_bound)
    pre_module = module.pre_nodes[0] if module.pre_nodes else None
    bound = None
    if (
        not in_residual_block
        and pre_module is not None
        and isinstance(pre_module, NonLinearNode)
        and pre_module.act_relax_args.update_scalar_bounds_per_layer
    ):
        logger.debug(f"Update scalar bounds by {pre_module}.")
        pre_scalar_bound = pre_module.all_bounds[pre_module.name].reshape(
            *constr_bound.L.A.shape[1:]
        )

        bound, _ = self.cal_bounds(constr_bound, pre_scalar_bound)
        if store_updated_bounds:
            bound = self.store_bounds(self.all_bounds, self.name, bound)

    return constr_bound, bound


def collect_residual_second_path(
    module: "BasicIneqNode",  # noqa
    constr_bound: LConstrBound,
    residual_second_path: list["BasicIneqNode"],  # noqa
) -> tuple[LConstrBound, list["BasicIneqNode"]]:  # noqa
    # For residual block, there will be two paths to the input.
    # We collect the modules in the second path and process them later.
    # The input biases will be calculated in the second path, so we do not
    # need the biases for the second path.
    constr_bound_r = constr_bound.detach().clone()
    constr_bound_r.L.b = None
    if constr_bound_r.U is not None:
        constr_bound_r.U.b = None
    # Store the modules in the second path.
    module_r = module.pre_nodes[1]
    while len(module_r.next_nodes) == 1:
        residual_second_path.append(module_r)
        module_r = module_r.pre_nodes[0]

    return constr_bound_r, residual_second_path


def back_substitute_residual_second_path(
    self: "BasicIneqNode",  # noqa
    module: "BasicIneqNode",  # noqa
    constr_bound: LConstrBound,
    constr_bound_r: LConstrBound,
    residual_second_path: list["BasicIneqNode"],  # noqa
    store_updated_bounds: bool = True,
) -> (
    tuple[LConstrBound, None, list["BasicIneqNode"]]  # noqa
    | tuple[LConstrBound, None, list["BasicIneqNode"], ScalarBound | None]  # noqa
):
    logger = logging.getLogger("rover")
    residual_second_path: list["BasicIneqNode"]  # noqa
    for module_r in residual_second_path:
        constr_bound_r = module_r.back_substitute_once(constr_bound_r)
    constr_bound_r.L.A = constr_bound_r.L.A.reshape(constr_bound.L.A.shape)
    if constr_bound_r.U is not None:
        constr_bound_r.U.A = constr_bound_r.U.A.reshape(constr_bound.U.A.shape)
    constr_bound = constr_bound + constr_bound_r
    # Reset
    constr_bound_r = None
    residual_second_path.clear()

    bound = None
    # Handle the non-linear module in the beginning of the residual block.
    if (
        isinstance(module, NonLinearNode)
        and module.act_relax_args.update_scalar_bounds_per_layer
    ):
        logger.debug(f"Update scalar bounds by {module}.")
        pre_scalar_bound = module.all_bounds[module.name].reshape(
            *constr_bound.L.A.shape[1:]
        )

        bound, _ = self.cal_bounds(constr_bound, pre_scalar_bound)
        if store_updated_bounds:
            bound = self.store_bounds(self.all_bounds, self.name, bound)

    return constr_bound, constr_bound_r, residual_second_path, bound
