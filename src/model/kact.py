"""
This module is used to build a linear programming model to verify the neural network
with k-activation constraints.

k-activation constraints is originally proposed in the paper:
`Beyond the single neuron convex barrier for neural network certification
<https://proceedings.neurips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf>`__
:cite:`singh_beyond_2019`

.. tip::

    In our terminology, k-activation constraints are also called multi-neuron
    constraints or k-act constraints.

We will use a technique called *lazy constraints* to add the k-act constraints when
enable this feature; otherwise, the k-act constraints will be added to the model
before solving the linear program.

.. tip::

    **Lazy constraints** is a technique to add constraints to the model when the
    solution violates the constraints.

    - First, we separate the k-act constraints from the original model and store them
      in a sparse matrix. Solving the linear program without the k-act constraints is
      faster but with a worse optimal solution.
    - When the optimal solution is not enough to verify the property, we will evaluate
      if adding the k-act constraints to the model can help to verify the property.
      That is, the original solution must close to what we want to verify.
    - Then, we filter those k-act constraints that are violated by the solution and add
      them to the model. This can improve the solution quality and help to verify the
      property.


"""

__docformat__ = "restructuredtext"
__all__ = ["KActLPBoundModel"]

import gurobipy
import logging
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
from gurobipy import GRB

from src.boundprop import *
from src.boundprop.kact import *
from src.linprog import *
from src.model.lp import LPBoundModel
from src.utils import *


class KActLPBoundModel(LPBoundModel):
    def __init__(
        self,
        net_file_path: str,
        perturbation_args: PerturbationArgs,
        act_relax_args: ActRelaxArgs,
        lp_args: LPArgs,
        kact_lp_args: KActLPArgs,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        """
        This class is used to build a linear programming model to verify the neural
        network with k-activation constraints.

        :param net_file_path: The path of the neural network file.
        :param perturbation_args: The perturbation arguments.
        :param act_relax_args: The activation relaxation arguments.
        :param multi_act_relax_args: The multi-neuron activation relaxation arguments.
        :param ada_act_relax_args: The adaptive constraints arguments.
        :param lp_args: The linear programming arguments.
        :param kact_lp_args: The k-activation arguments.
        :param log_args: The logger arguments.
        :param dtype: The data type of the linear program.
        :param device: The device of the linear program.
        """
        LPBoundModel.__init__(
            self,
            net_file_path,
            perturbation_args,
            act_relax_args,
            lp_args=lp_args,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.kact_lp_args = kact_lp_args

        self._kact_constrs_A = None
        self._kact_constrs_RHS = None
        self._kact_constr_counter = 0

        self._has_built_kact_constrs = False

    def verify_lp(self, label: int, adv_labels: list[int] = None) -> list[bool]:
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

            success, obj_val, solution = self.solve_lp(
                call_back=False, return_solution=True
            )
            if (
                obj_val is not None
                and 0 > obj_val > self.kact_lp_args.gurobi_lazy_callback_objval
            ):
                if not self._has_built_kact_constrs:
                    self.build_kact_lp()
                if self.kact_lp_args.use_lazy_constraints:
                    self._add_violated_kact_constrs(solution)
                self.model.update()
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

        .. tip::
            When lazy constraints are used, the callback function use a negative value
            to evaluate if continue to solve the linear program with adding lazy
            constraints. If the objective value is less than the negative value, the
            linear program will be terminated without adding lazy constraints.


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

        def slow_callback_lp(model: gurobipy.Model, where: GRB.Callback):
            if where == GRB.Callback.SIMPLEX:
                obj_best = model.cbGet(GRB.Callback.SPX_OBJVAL)
                if (
                    model.cbGet(GRB.Callback.SPX_PRIMINF) == 0
                    and obj_best < self.kact_lp_args.gurobi_lazy_callback_objval
                ):
                    # and model.cbGet(GRB.Callback.SPX_DUALINF) == 0:
                    model.terminate()

        def callback_lp(model: gurobipy.Model, where: GRB.Callback):
            if where == GRB.Callback.SIMPLEX:
                obj_best = model.cbGet(GRB.Callback.SPX_OBJVAL)
                if model.cbGet(GRB.Callback.SPX_PRIMINF) == 0 and obj_best < -0.001:
                    # and model.cbGet(GRB.Callback.SPX_DUALINF) == 0:
                    model.terminate()

        if call_back:
            self.model.optimize(callback_lp)
        else:
            self.model.optimize(slow_callback_lp)

        result, obj_val, solution = self._process_model_status(return_solution)

        logger.info(f"Finish solving LP model in {time.perf_counter() - start:.4f}s")
        logger.info(f"Verification result: {'SUCCESS' if result else 'UNKNOWN'}.")

        return result, obj_val, solution

    def build_kact_lp(self):
        """
        Build the multi-neuron constraints for the linear program.


        """
        logger = logging.getLogger("rover")
        logger.info("Start building KAct constraints.")
        time_start = time.perf_counter()

        input_bound = self.all_bounds[self.input_name]

        for module in self.submodules.values():
            if not isinstance(module, NonLinearNode):
                continue

            pre_module = module.pre_nodes[0]
            pre_bound = self.all_bounds[pre_module.name]
            pool_input_ids = None
            pool_input_l = None
            pool_input_u = None
            if isinstance(module, ReLUNode):
                act_type = ActivationType.RELU
            elif isinstance(module, SigmoidNode):
                act_type = ActivationType.SIGMOID
            elif isinstance(module, TanhNode):
                act_type = ActivationType.TANH
            elif isinstance(module, LeakyReLUNode):
                act_type = ActivationType.LEAKY_RELU
            elif isinstance(module, ELUNode):
                act_type = ActivationType.ELU
            elif isinstance(module, MaxPool2DNode):
                act_type = ActivationType.MAXPOOL2D

                n_c, n_h, n_w = module.input_size  # number of channels
                n_k = math.prod(module.output_size[1:])  # number of output elements
                n_kx = math.prod(module.kernel_size)  # number of kernel elements

                kwargs = {
                    "kernel_size": module.kernel_size,
                    "dilation": module.dilation,
                    "padding": module.padding,
                    "stride": module.stride,
                }

                # The following operation does not support integers.
                pool_input_ids = (
                    F.unfold(
                        torch.arange(
                            math.prod(module.input_size), **self.data_settings
                        ).reshape(module.input_size),
                        **kwargs,
                    )
                    .to(dtype=torch.long)  # shape=(n_c*n_kx, :)
                    .reshape(n_c, n_kx, -1)
                    .transpose(1, 2)
                    .reshape(-1, n_kx)
                )

                pool_input_l = (
                    F.unfold(pre_bound.l.reshape(n_c, n_h, n_w), **kwargs)
                    .reshape((n_c, n_kx, n_k))
                    .permute(0, 2, 1)
                    .reshape((n_c * n_k, n_kx))
                )
                pool_input_u = (
                    F.unfold(pre_bound.u.reshape(n_c, n_h, n_w), **kwargs)
                    .reshape((n_c, n_kx, n_k))
                    .permute(0, 2, 1)
                    .reshape((n_c * n_k, n_kx))
                )
            else:
                raise NotImplementedError(f"{module} is not supported.")

            logger.info(f"Start building KAct constraints for {module}.")

            if isinstance(module, MaxPool2DNode):
                pre_bound = ScalarBound(*module.get_unfolded_pre_bound(pre_bound))

            mask_mn = module.get_nontrivial_neuron_mask(
                pre_bound,
                cached=False,
                recalculate=True,
                ignore_degenerate_pool=False,
            )
            n_mn = mask_mn.sum().item()
            if n_mn < self.kact_lp_args.group_size:
                logger.info(f"To few neurons {n_mn} to group for {module}.")
                continue

            grouped_input_ids = generate_groups_lp(
                pre_bound.l,
                pre_bound.u,
                mask_mn,
                act_type,
                self.kact_lp_args,
                pool_input_ids=pool_input_ids,
            )

            grouped_input_constrs = back_substitute_grouped_constrs(
                input_bound,
                pre_module,
                pre_bound.l.numel(),
                grouped_input_ids,
                self.kact_lp_args.max_parallel_groups,
                self.kact_lp_args.constr_template,
                **self.data_settings,
            )

            grouped_output_constrs = cal_grouped_acthull(
                pre_bound.l,
                pre_bound.u,
                mask_mn,
                grouped_input_ids,
                grouped_input_constrs,
                act_type,
                pool_input_l=pool_input_l,
                pool_input_u=pool_input_u,
                use_multi_threads=self.kact_lp_args.use_multi_threads,
            )

            grouped_input_ids = grouped_input_ids.tolist()

            if not isinstance(module, MaxPool2DNode):
                grouped_output_ids = grouped_input_ids
            else:
                grouped_output_ids = torch.arange(
                    math.prod(module.output_size), device=mask_mn.device
                )[mask_mn.flatten()].tolist()

            lp_module = self.lp_submodules[module.name]
            lp_module: NonLinearLPNode
            lp_module.add_kact_constrs(
                self.all_vars[pre_module.name],
                self.all_vars[module.name],
                grouped_input_ids,
                grouped_output_ids,
                grouped_output_constrs,
                self.model,
            )

            self.model.update()

        logger.info(
            f"Finish building KAct constraints in "
            f"{time.perf_counter() - time_start:.4f}s"
        )
        logger.info(
            f"Current LP model has {self.model.NumVars} variables "
            f"and {self.model.NumConstrs} constraints"
        )

        if self.kact_lp_args.use_lazy_constraints:
            self._separate_kact_constrs()

        self._has_built_kact_constrs = True

    def _separate_kact_constrs(self):
        """
        Separate the KAct constraints from the original model.

        .. tip::
            The k-act constraints are separated from the original model to reduce the
            size of the model and so a faster solution can be obtained.

            The k-act constraints will be added to the model when the solution
            violates the constraints, where we also assess if there is a need to add
            k-act constraints to the model. This is called *lazy constraints*.

        :return:
        """

        logger = logging.getLogger("rover")
        logger.info("Start Separating KAct constraints.")
        time_start = time.perf_counter()

        # Get all kact constraints from the original model
        model_with_act_constrs: gurobipy.Model = self.model.copy()
        for constr in model_with_act_constrs.getConstrs():
            if not constr.getAttr("ConstrName").endswith("kact"):
                model_with_act_constrs.remove(constr)
        model_with_act_constrs.update()

        # The matrix as a scipy.sparse matrix in CSR format.
        self._kact_constrs_A = model_with_act_constrs.getA()
        self._kact_constrs_RHS = np.array(
            model_with_act_constrs.getAttr("RHS"), dtype=np.float64
        )

        logger.debug(f"KAct constraints number: {self._kact_constrs_A.shape[0]}")

        act_constrs_sense = model_with_act_constrs.getAttr("Sense")
        if any(sense != ">" for sense in act_constrs_sense):
            raise ValueError("The sense of act constraints should be '>'.")

        model_with_act_constrs.dispose()

        # Remove all kact constraints from the original model
        for constr in self.model.getConstrs():
            if constr.getAttr("ConstrName").endswith("kact"):
                self.model.remove(constr)

        self.model.update()
        self._has_separate_kact_constrs = True
        self._kact_constr_counter = 0

        logger.info(
            f"Finish separating KAct constraints in "
            f"{time.perf_counter() - time_start:.4f}s"
        )

    def _add_violated_kact_constrs(self, solution: np.ndarray):
        logger = logging.getLogger("rover")
        if self._kact_constrs_A.shape[0] == 0:
            logger.info("No KAct constraints to add.")
            return

        logger.info(f"Start adding violated KAct constraints.")
        time_start = time.perf_counter()

        def add_kact_constrs(constrs_A, constrs_RHS: np.ndarray):
            vars_ = np.asarray([var for var in self.model.getVars()])
            for i in range(constrs_A.shape[0]):
                m, b = constrs_A[i], constrs_RHS.item(i)
                mx = gurobipy.LinExpr(
                    [m[0, c] for c in m.indices], [vars_[c] for c in m.indices]
                )
                self.model.addLConstr(
                    lhs=mx,
                    sense=GRB.GREATER_EQUAL,
                    rhs=b,
                    name=f"kact_{self._kact_constr_counter}",
                )
                self._kact_constr_counter += 1
            self.model.update()

        value = self._kact_constrs_A @ solution - self._kact_constrs_RHS
        violated_constrs_idxes = np.where(value < 0)[0]
        logger.info(
            f"Add {violated_constrs_idxes.shape[0]}/{self._kact_constrs_A.shape[0]} "
            f"violated KAct constraints."
        )
        add_kact_constrs(
            self._kact_constrs_A[violated_constrs_idxes],
            self._kact_constrs_RHS[violated_constrs_idxes],
        )

        remained_constrs_idxes = np.ones(self._kact_constrs_A.shape[0], dtype=np.bool_)
        remained_constrs_idxes[violated_constrs_idxes] = False
        self._kact_constrs_A = self._kact_constrs_A[remained_constrs_idxes]
        self._kact_constrs_RHS = self._kact_constrs_RHS[remained_constrs_idxes]

        logger.info(
            f"Finish adding violated KAct constraints in "
            f"{time.perf_counter() - time_start:.4f}s"
        )
        logger.info(
            f"Current LP model has {self.model.NumVars} variables "
            f"and {self.model.NumConstrs} constraints"
        )

    def clear(self):
        """
        Clear the linear programming model. Specifically,

        - clear the Gurobi model with model.close() and
        - clear the shared data including all Gurobi variables and constraints.
        - clear the cached k-act constraints.

        """
        super().clear()

        logger = logging.getLogger("rover")
        logger.debug("Clear cache of kact constraints.")

        self._kact_constrs_A = None
        self._kact_constrs_RHS = None
        self._kact_constr_counter = 0

        self._has_built_kact_constrs = False
