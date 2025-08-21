"""
This module is used to build a linear programming model to verify the neural network.
This module is based on a bound propagation method to calculate the scalar bounds of
neurons. By these bounds of neurons, linear constraints are constructed serve to the
linear program. The property to verify is a linear form, and it is taken as the
objective function of the linear program. The linear program is solved by Gurobi.
"""

__docformat__ = "restructuredtext"
__all__ = ["LPBoundModel", "GRB_STATUS_MAP"]

import logging
import time
from collections import defaultdict, OrderedDict
from typing import Generic, TypeVar

import gurobipy
import numpy as np
import torch
from gurobipy import GRB
from torch import Tensor

from src.boundprop import *
from src.linprog import *
from src.model.ineq import IneqBoundModel
from src.utils import *

GRB_STATUS_MAP = defaultdict(str)
GRB_STATUS_MAP.update(
    {
        GRB.LOADED: "LOADED",  # 1
        GRB.OPTIMAL: "OPTIMAL",  # 2
        GRB.INFEASIBLE: "INFEASIBLE",  # 3
        GRB.INF_OR_UNBD: "INF_OR_UNBD",  # 4
        GRB.UNBOUNDED: "UNBOUNDED",  # 5
        GRB.CUTOFF: "CUTOFF",  # 6
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",  # 7
        GRB.NODE_LIMIT: "NODE_LIMIT",  # 8
        GRB.TIME_LIMIT: "TIME_LIMIT",  # 9
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",  # 10
        GRB.INTERRUPTED: "INTERRUPTED",  # 11
        GRB.NUMERIC: "NUMERIC",  # 12
        GRB.SUBOPTIMAL: "SUBOPTIMAL",  # 13
        GRB.INPROGRESS: "INPROGRESS",  # 14
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",  # 15
        GRB.WORK_LIMIT: "WORK_LIMIT",  # 16
        GRB.MEM_LIMIT: "MEM_LIMIT",  # 17
    }
)

T = TypeVar("T", bound=BasicLPNode)

_TOLERANCE = 1e-6


class LPBoundModel(IneqBoundModel, Generic[T]):
    def __init__(
        self,
        net_file_path: str,
        perturbation_args: PerturbationArgs,
        act_relax_args: ActRelaxArgs,
        lp_args: LPArgs,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        """
        This class is used to build a linear programming model to verify the neural
        network. This module is based on a bound propagation method to calculate the
        scalar bounds of neurons. By these bounds of neurons, linear constraints are
        constructed serve to the linear program. The property to verify is a linear
        form, and it is taken as the objective function of the linear program. The
        linear program is solved by Gurobi.

        :param net_file_path: The path of the neural network file.
        :param perturbation_args: The perturbation arguments.
        :param act_relax_args: The activation relaxation arguments.
        :param multi_act_relax_args: The multi-neuron activation relaxation arguments.
        :param ada_act_relax_args: The adaptive constraints arguments.
        :param lp_args: The linear programming arguments.
        :param log_args: The logger arguments.
        :param dtype: The data type of the linear program.
        :param device: The device of the linear program.
        """
        IneqBoundModel.__init__(
            self,
            net_file_path,
            perturbation_args,
            act_relax_args,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.lp_args = lp_args

        self.model: gurobipy.Model | None = None
        self.lp_submodules: dict[str, T] = OrderedDict()
        self.lp_shared_data: LPSharedData = LPSharedData()
        self.output_vars: list[gurobipy.Var] | None = None

    def _init_gurobi_model(self):

        model = gurobipy.Model()
        model.setParam(GRB.Param.OutputFlag, self.lp_args.gurobi_output_flag)
        model.setParam(GRB.Param.OptimalityTol, self.lp_args.gurobi_optimality_tol)
        model.setParam(GRB.Param.FeasibilityTol, self.lp_args.gurobi_feasibility_tol)
        model.setParam(GRB.Param.NumericFocus, self.lp_args.gurobi_numeric_focus)
        model.setParam(GRB.Param.MarkowitzTol, self.lp_args.gurobi_markowitz_tol)
        model.setParam(GRB.Param.Method, self.lp_args.gurobi_method)
        model.setParam(GRB.Param.TimeLimit, self.lp_args.gurobi_timelimit)
        model.setParam(GRB.Param.Cutoff, self.lp_args.gurobi_cutoff)

        return model

    def build(
        self,
        output_weight: Tensor | None = None,
        output_bias: Tensor | None = None,
    ):
        """
        Build the linear programming module from the ONNX model file.

        :param output_weight: The output weight matrix.
        :param output_bias: The output bias.
        """
        logger = logging.getLogger("rover")

        super().build(output_weight, output_bias)

        logger.debug("Start building LP module.")
        time_start = time.perf_counter()

        def convert_to_lp_module(m: BasicNode) -> BasicLPNode:
            lp_m = None
            args = (
                module.name,
                module.input_names,
                module.input_size,
                module.output_size,
                self.lp_shared_data,
                self.lp_args,
            )
            if isinstance(m, InputNode):
                lp_m = InputLPNode(*args)

            elif isinstance(m, ResidualAddNode):
                # It is also a linear node, so put it before the LinearNode.
                lp_m = ResidualAddLPNode(*args)

            elif isinstance(m, LinearNode):

                weight = m.weight.tolist()
                bias = None if m.bias is None else m.bias.tolist()

                if isinstance(m, GemmNode):
                    lp_m = GemmLPNode(*args, weight=weight, bias=bias)
                elif isinstance(m, Conv2DNode):
                    lp_m = Conv2DLPNode(
                        *args,
                        weight=weight,
                        bias=bias,
                        stride=m.stride,
                        padding=m.padding,
                        dilation=m.dilation,
                        groups=m.groups,
                        ceil_mode=m.ceil_mode,
                    )

            elif isinstance(m, NonLinearNode):
                act_relax_args = m.act_relax_args
                if isinstance(m, ReLUNode):
                    lp_m = ReLULPNode(*args, act_relax_args=act_relax_args)
                elif isinstance(m, SigmoidNode):
                    lp_m = SigmoidLPNode(*args, act_relax_args=act_relax_args)
                elif isinstance(m, TanhNode):
                    lp_m = TanhLPNode(*args, act_relax_args=act_relax_args)
                elif isinstance(m, ELUNode):
                    lp_m = ELULPNode(*args, act_relax_args=act_relax_args)
                elif isinstance(m, LeakyReLUNode):
                    lp_m = LeakyReLULPNode(*args, act_relax_args=act_relax_args)
                elif isinstance(m, MaxPool2DNode):
                    lp_m = MaxPool2DLPNode(
                        *args,
                        act_relax_args=act_relax_args,
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                        dilation=m.dilation,
                        ceil_mode=m.ceil_mode,
                    )

                else:
                    raise NotImplementedError(f"Module {m} is not supported.")

            else:
                raise NotImplementedError(f"Module {m} is not supported.")

            return lp_m

        # Convert all modules to LP modules
        for module in self.submodules.values():
            logger.debug(
                f"Create LP module of {module.__class__.__name__}({module.name})."
            )
            lp_module = convert_to_lp_module(module)
            self.lp_submodules[lp_module.name] = lp_module

        # Set the pre and next nodes for each module
        logger.debug("Set pre and next nodes for each LP module.")
        for module in self.submodules.values():
            lp_module = self.lp_submodules[module.name]

            if module.pre_nodes is not None:
                lp_module.pre_nodes = []
                for pre_module in module.pre_nodes:
                    lp_pre_module = self.lp_submodules[pre_module.name]
                    lp_module.pre_nodes.append(lp_pre_module)
            else:
                # The input module has no pre nodes.
                lp_module.pre_nodes = None

            logger.debug(
                f"Set pre nodes of {module.__class__.__name__}({module.name}) "
                f"being {lp_module.pre_nodes}."
            )

            if module.next_nodes is not None:
                lp_module.next_nodes = []
                for next_module in module.next_nodes:
                    lp_next_module = self.lp_submodules[next_module.name]
                    lp_module.next_nodes.append(lp_next_module)
            else:
                # The output module has no next nodes.
                lp_module.next_nodes = None

            logger.debug(
                f"Set next nodes of {module.__class__.__name__}({module.name}) "
                f"being {lp_module.next_nodes}."
            )

        logger.info(
            f"Finish building LP module in {time.perf_counter() - time_start:.4f}s."
        )

    def build_lp(self):
        logger = logging.getLogger("rover")

        logger.debug("Start building LP model.")
        time_start = time.perf_counter()

        logger.debug("Clear all parameters from bound propagation if any.")
        self.bp_shared_data.clear_params()

        logger.debug("Restore last weights and biases.")
        self.restore_output_constraints()

        logger.debug("Initialize Gurobi model.")
        self.model = self._init_gurobi_model()

        gvars = None
        for module, lp_module in zip(
            self.submodules.values(), self.lp_submodules.values()
        ):
            logger.debug(
                f"Add variables and constraints for {lp_module.__class__.__name__} "
                f"({lp_module.name})."
            )
            # Calculate the scalar bounds for non-linear modules.
            bound = self.all_bounds.get(lp_module.name, None)  # type: ignore

            if bound is None:

                if not isinstance(module, NonLinearNode):
                    raise RuntimeError(f"Bound for {module.name} is not calculated.")

                pre_bound = self.all_bounds[module.input_names[0]]

                if not isinstance(module, MaxPool2DNode):
                    l = module.f(pre_bound.l)
                    u = module.f(pre_bound.u)
                else:
                    # For MaxPool2D, we take the largest lower bound and the largest
                    # upper bound of the pooling region.
                    l = module.f(pre_bound.l.reshape(module.input_size))
                    u = module.f(pre_bound.u.reshape(module.input_size))

                bound = ScalarBound(l=l, u=u)
                self.all_bounds[module.name] = bound

            if lp_module.next_nodes is None:
                # For the output layer, we set the bound be None.
                # Because the bound provided by bound propagation is only the lower
                # bound of the merged layer, we cannot use it.
                bound = None

            if bound is not None:
                if not torch.all(bound.l <= bound.u + _TOLERANCE):
                    mask = bound.l > bound.u
                    raise RuntimeError(
                        f"For {lp_module.__class__.__name__}, \n"
                        f"The lower bound is greater than the upper bound. \n"
                        f"Lower bound: {bound.l[mask]}, \n"
                        f"Upper bound: {bound.u[mask]}."
                    )

            gvars = lp_module.add_vars(bound, self.model)
            self.all_vars[lp_module.name] = gvars  # type: ignore

            if len(module.input_names) == 0:
                pass

            elif len(lp_module.input_names) == 1:  # type: ignore
                pre_name = lp_module.input_names[0]  # type: ignore
                self.all_constrs[lp_module.name] = lp_module.add_constrs(  # noqa
                    gvars,
                    self.model,
                    self.all_vars[pre_name],
                    self.all_bounds[pre_name],
                )

            elif len(lp_module.input_names) == 2:  # type: ignore
                pre_name1, pre_name2 = lp_module.input_names
                self.all_constrs[lp_module.name] = lp_module.add_constrs(  # noqa
                    gvars,
                    self.model,
                    self.all_vars[pre_name1],
                    self.all_bounds[pre_name1],
                    self.all_vars[pre_name2],  # type: ignore
                    self.all_bounds[pre_name2],  # type: ignore
                )

            else:
                raise RuntimeError(
                    f"{lp_module} has more than 2 inputs (" f"{lp_module.input_names})."
                )

        self.model.update()
        self.output_vars = gvars
        logger.debug(
            f"Finish building LP model in {time.perf_counter() - time_start:.4f}s"
        )
        logger.info(
            f"Current LP model has {self.model.NumVars} variables "
            f"and {self.model.NumConstrs} constraints"
        )

        # # The following code is to check if the mode is infeasible.
        # logger.debug(f"Check if the model is infeasible.")
        # self.model.optimize()
        # if self.model.Status == GRB.INFEASIBLE:
        #     logger.error(f"The model is infeasible. Write the model to model.ilp")
        #     self.model.computeIIS()
        #     # Create a directory to store the model
        #     os.makedirs("ilp", exist_ok=True)
        #     self.model.write("ilp/model.ilp")
        #     raise RuntimeError(f"The model is infeasible.")
        # else:
        #     logger.debug(
        #         f"The model is feasible {self.model.Status}. "
        #         "The objective value is {self.model.ObjVal:.4f}"
        #     )

    def restore_output_constraints(self):
        """
        Restore the original weight and bias of the last layer, i.e., remove the
        output weight matrix and bias from the merged layer.

        .. tip::
            If the original model does not have a last layer as linear layer, we need to
            create an identity matrix as the output weight matrix and remove the bias.

        .. hint::
            The reason to restore the output constraints is that we do not need to
            change the gurobi model's constraints but change the objective function.

        .. attention::
            Currently, this method only supports local robustness verification.
        """

        logger = logging.getLogger("rover")

        last_module = self.submodules[self.output_name]
        if not isinstance(last_module, GemmNode):
            raise RuntimeError(f"Unsupported last layer type {type(last_module)}.")

        if self._ori_last_weight is not None and self._ori_last_bias is not None:
            last_module.weight = self._ori_last_weight
            last_module.bias = self._ori_last_bias
            logger.debug("Restore output constraints for the last layer.")
        else:
            # Here the implementation is DIFFERENT from the superclass.
            n = last_module.weight.shape[0]
            last_module.weight = torch.eye(n, dtype=self.dtype, device=self.device)
            last_module.bias = None
            logger.debug("No need to restore the output constraints.")

    def verify_lp(self, label: int, adv_labels: list[int] = None) -> list[bool]:
        """
        Verify the property of the neural network by solving the linear program. This is
        for verifying local robustness.

        .. attention::
            Currently, this method only supports local robustness verification.

        :param label: The label to verify.
        :param adv_labels: The adversarial labels.

        :return: The verification results, which is a list of boolean values, where True
            means the property is verified, and False means the property is not
            verified.
        """
        logger = logging.getLogger("rover")

        num_labels = len(self.output_vars)
        if adv_labels is None:
            adv_labels = list(range(num_labels))
            adv_labels.pop(label)

        results = [
            False if label in adv_labels else True for label in range(num_labels)
        ]

        for adv_label in adv_labels:
            logger.info(f"Verify label {adv_label} vs. {label}.")

            obj = self.output_vars[label] - self.output_vars[adv_label]
            self.model.setObjective(obj)

            success, _, _ = self.solve_lp()

            results[adv_label] = success

            if not success and self.lp_args.terminate_if_fail:
                break

        return results

    def solve_lp(
        self, call_back=False, return_solution: bool = False
    ) -> tuple[bool, float | None, np.ndarray | None]:
        """
        Solve the linear program.

        :param call_back: Whether to use the callback function.
        :param return_solution: Whether to return the solution.

        .. tip::

            Normally, the linear program should not be *infeasible*. If the linear
            program is infeasible, the infeasible constraints will be output to a
            file for debug.

            The reasons for infeasibility may be:

            - There is a bug when calculating the bounds and relaxation of neurons.
            - There is a bug when constructing the linear program.
            - The numerical precision of the solver is not enough for tiny bounds and
              constraints.

        :return: A tuple of three elements, where the first element is a boolean value
            indicating if the linear program successfully verifies the property, the
            second element is the objective value, and the third element is the
            solution.

            - If the linear program has not got the optimal solution, the
              objective value and the solution will be None.
            - If return_solution is False, the solution will be None.
        """
        logger = logging.getLogger("rover")
        logger.info("Start solving LP model.")
        start = time.perf_counter()

        def callback_lp(model: gurobipy.Model, where: GRB.Callback):
            if where == GRB.Callback.SIMPLEX:
                obj_best = model.cbGet(GRB.Callback.SPX_OBJVAL)
                if model.cbGet(GRB.Callback.SPX_PRIMINF) == 0 and obj_best < -0.001:
                    # and model.cbGet(GRB.Callback.SPX_DUALINF) == 0:
                    model.terminate()

        if call_back:
            self.model.optimize(callback_lp)
        else:
            self.model.optimize()

        result, obj_val, solution = self._process_model_status(return_solution)

        logger.info(f"Finish solving LP model in {time.perf_counter() - start:.4f}s")
        logger.info(f"Verification result: {'SUCCESS' if result else 'UNKNOWN'}")

        return result, obj_val, solution

    def _process_model_status(
        self, return_solution: bool = False
    ) -> tuple[bool, float | None, np.ndarray | None]:
        logger = logging.getLogger("rover")

        logger.info(
            f"Result status: {self.model.Status}-"
            f"{GRB_STATUS_MAP.get(self.model.status)}."
        )

        # If the model is infeasible or unbounded, reoptimize to get definitive status.
        if self.model.Status == GRB.INF_OR_UNBD:
            logger.warning(
                "The model is infeasible or unbounded. "
                "Reoptimize to get definitive status."
            )
            self.model.setParam(GRB.Param.DualReductions, 0)
            self.model.optimize()
            logger.info(
                f"Result status: {self.model.Status}-"
                f"{GRB_STATUS_MAP.get(self.model.status)}."
            )

        # If the model is infeasible, output the infeasible constraints.
        if self.model.Status == GRB.INFEASIBLE:
            # Set the numeric focus to 3 to get higher precision.
            logger.info(
                "The model is infeasible. Set numeric focus to 3 and recompute."
            )
            self.model.setParam(GRB.Param.NumericFocus, 3)
            self.model.optimize()
            logger.info(
                f"Result status: {self.model.Status}-"
                f"{GRB_STATUS_MAP.get(self.model.status)}."
            )

            if self.model.Status == GRB.INF_OR_UNBD:
                logger.warning(
                    "The model is infeasible or unbounded. "
                    "Reoptimize to get definitive status."
                )
                self.model.setParam(GRB.Param.DualReductions, 0)
                self.model.optimize()
                logger.info(
                    f"Result status: {self.model.Status}-"
                    f"{GRB_STATUS_MAP.get(self.model.status)}."
                )

            if self.model.Status == GRB.INFEASIBLE:
                logger.info("The model is still infeasible.")
                current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
                file_name = f"infeasible_model_IIS_{current_time}.ilp"
                try:
                    self.model.computeIIS()
                    self.model.write(file_name)
                    logger.error(f"The infeasible model IIS is written to {file_name}.")
                    # logger.error(f"The model is infeasible.")
                except Exception as e:
                    logger.error(f"Failed to write the infeasible model IIS: {e}")

        # If the model is optimal, get the objective value and solution.
        result = self.if_lp_success()
        obj_val = None
        solution = None
        if self.model.Status == GRB.OPTIMAL:
            obj_val = self.model.ObjVal
            logger.info(f"Objective value is {obj_val:.4f}.")
            if return_solution:
                solution = np.array(
                    [var.x for var in self.model.getVars()], dtype=np.float64  # noqa
                )

        return result, obj_val, solution

    def if_lp_success(self) -> bool:
        """
        Check if the linear program successfully verify the property.

        .. tip::
            Only two cases are considered as successful verification:

            1. The linear program is *optimal*, and the objective value is greater
               than 0.
            2. The linear program is *cutoff*, which means the solver has found a
               positive lower bound of the objective value

        """
        status = self.model.Status

        if status == GRB.OPTIMAL and self.model.ObjVal > 0.0:
            return True
        elif status == GRB.CUTOFF:
            return True

        return False

    def clear(self):
        """
        Clear the linear programming model. Specifically,

        - clear the Gurobi model with model.close() and
        - clear the shared data including all Gurobi variables and constraints.
        """
        super().clear()

        logger = logging.getLogger("rover")
        logger.debug("Clear cache of linear programming model.")

        self.lp_shared_data.clear()
        if self.model is not None:
            self.model.close()
        self.model = None
        self.output_vars: list[gurobipy.Var] | None = None

    @property
    def all_vars(self):
        """All Gurobi variables in the linear program."""
        return self.lp_shared_data.all_vars

    @property
    def all_constrs(self):
        """All Gurobi constraints in the linear program."""
        return self.lp_shared_data.all_constrs

    @property
    def all_lower_bounds(self):
        """All lower bounds of neurons."""
        return self.lp_shared_data.all_lower_bounds

    @property
    def all_upper_bounds(self):
        """All upper bounds of neurons."""
        return self.lp_shared_data.all_upper_bounds
