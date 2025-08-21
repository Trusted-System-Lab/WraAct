"""
This module contains the classes of the linear programming nodes.

Implementation Overview
------------------------

The linear programming nodes are mainly used to build the Gurobi variable and
constraints for the Gurobi model.
The information they based on are the result from bound propagation node, so the nodes
between the bound propagation node and the linear programming node have a one-to-one
correspondence.

Therefore, the main methods of these linear programming nodes are:

- `add_vars`: Add Gurobi variables to the Gurobi model based on the scalar bound of the
    variables.
- `add_constrs`: Add Gurobi constraints to the Gurobi model based on the relaxation for
    specific layer.

.. tip::
    The Gurobi has its inner data structure to store the variables and constraints,
    which is sparse, so there is some inconsistency between the data structure of the
    bound propagation nodes.
    Currently, some computation is repeated, which can be optimized in the future.

Data Format
~~~~~~~~~~~

By default, we only use "greater equal" and "equal" constraints for consistency to
convenient the further operations.

"""

__docformat__ = "restructuredtext"
__all__ = [
    "BasicLPNode",
    "InputLPNode",
    "LinearLPNode",
    "GemmLPNode",
    "Conv2DLPNode",
    "NonLinearLPNode",
    "ReLULPNode",
    "LeakyReLULPNode",
    "ELULPNode",
    "SShapeLPNode",
    "SigmoidLPNode",
    "TanhLPNode",
    "MaxPool2DLPNode",
    "ResidualAddLPNode",
]

import itertools
import logging
from abc import ABC, abstractmethod

import gurobipy
import math
import numpy as np

from src.linprog.containers import LPSharedData
from src.utils import *


class BasicLPNode(ABC):
    """
    Basic class of the linear programming node.

    :param name: The name of the node.
    :param input_names: The names of the input nodes.
    :param input_size: The size of the input data.
    :param output_size: The size of the output data.
    :param shared_data: The shared data between the nodes.
    :param lp_args: The arguments of the linear programming.
    :param kact_lp_args: The arguments of the k-activation linear programming.
    """

    _layer_count: int = 0
    _kact_layer_count: int = 0

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        self._input_names = input_names
        self._name = name
        self._input_size = input_size
        self._output_size = output_size
        self._pre_nodes: list["BasicLPNode"] | None = None
        self._next_nodes: list["BasicLPNode"] | None = None

        self._shared_data = shared_data

        self._lp_args = lp_args
        self._kact_lp_args = kact_lp_args

        BasicLPNode._layer_count += 1

    def add_vars(
        self,
        bound: ScalarBound | None,
        model: gurobipy.Model,
    ) -> list[gurobipy.Var]:
        """
        Add gurobi variables to the gurobi model based on the scalar bounds of the
        variables.

        .. tip::
            For the output variables, the bound is not provided, so the default bound
            is (-inf, inf).

        :param bound: Scalar bound of the variables.
        :param model: Gurobi model.

        :return: A list of gurobi variables.
        """
        logger = logging.getLogger("rover")

        if bound is not None:
            lower_bounds = bound.l.flatten().tolist()
            upper_bounds = bound.u.flatten().tolist()

            gvars = [
                model.addVar(
                    lb=l,
                    ub=u,
                    obj=0.0,
                    vtype=gurobipy.GRB.CONTINUOUS,
                    name=self._create_var_name(i),
                    column=None,
                )
                for i, (l, u) in enumerate(zip(lower_bounds, upper_bounds))
            ]
        else:
            gvars = [
                model.addVar(
                    lb=-gurobipy.GRB.INFINITY,
                    ub=gurobipy.GRB.INFINITY,
                    obj=0.0,
                    vtype=gurobipy.GRB.CONTINUOUS,
                    name=self._create_var_name(i),
                    column=None,
                )
                for i in range(int(math.prod(self.output_size)))
            ]

        logger.debug(f"Add {len(gvars)} variables to the model.")

        return gvars

    @abstractmethod
    def add_constrs(
        self,
        gvars: list[gurobipy.Var],
        model: gurobipy.Model,
        pre_gvars: list[gurobipy.Var] | None = None,
        pre_bound: ScalarBound | None = None,
    ) -> list[gurobipy.Constr]:
        """
        Add gurobi constraints to the gurobi model based on the relaxation for specific
        layer.

        .. attention::
            Currently, we repeatedly calculate the linear relaxation for the
            constraints, because the linear programming can accept more constraints
            rather than two constraints for the lower and upper bounds.
            The current implementation maybe optimal, because the parallel computation
            is not used.
            Maybe we can optimize the implementation in the future to realize parallel
            computing and adding constraints.
            But this obtained efficiency may be limited, because this only take a small
            part of the whole optimization process.

        :param gvars: Gurobi variables of the current layer.
        :param model: Gurobi model.
        :param pre_gvars: Gurobi variables of the preceding layer.
        :param pre_bound: Scalar bound of the variables of preceding layer.

        :return: A list of gurobi constraints.
        """
        pass

    @property
    def input_names(self) -> list[str]:
        """Names of the input nodes."""
        return self._input_names

    @input_names.setter
    def input_names(self, names: list[str]):
        """Set the names of the input nodes."""
        self._input_names = names

    @property
    def name(self) -> str:
        """Name of the node."""
        return self._name

    @name.setter
    def name(self, name: str):
        """Set the name of the node."""
        self._name = name

    @property
    def input_size(self) -> tuple[int] | tuple[int, int, int]:
        """Size of the input data."""
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int] | tuple[int, int, int]):
        """Set the size of the input data."""
        self._input_size = size

    @property
    def output_size(self) -> tuple[int] | tuple[int, int, int]:
        """Size of the output data."""
        return self._output_size

    @output_size.setter
    def output_size(self, size: tuple[int] | tuple[int, int, int]):
        """Set the size of the output data."""
        self._output_size = size

    @property
    def pre_nodes(self) -> list["BasicLPNode"]:
        """Preceding nodes."""
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, nodes: list["BasicLPNode"]):
        """Set the preceding nodes."""
        self._pre_nodes = nodes

    @property
    def next_nodes(self) -> list["BasicLPNode"]:
        """Next nodes."""
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, node: list["BasicLPNode"]):
        """Set the next nodes."""
        self._next_nodes = node

    def _create_var_name(self, idx: int) -> str:
        """Create the name of the variable."""
        return f"{self.__class__.__name__}_{self.name}_{idx}"

    def _create_constr_name(self, idx: int) -> str:
        """Create the name of the constraint."""
        return f"{self.__class__.__name__}_{self.name}_{idx}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.name})"

    def __repr__(self) -> str:
        return self.__str__()


class InputLPNode(BasicLPNode):
    """
    A class of the linear programming node for the input layer.

    :param name: The name of the node.
    :param input_names: The names of the input nodes.
    :param input_size: The size of the input data.
    :param output_size: The size of the output data.
    :param shared_data: The shared data between the nodes.
    :param lp_args: The arguments of the linear programming.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
    ):
        BasicLPNode.__init__(
            self, name, input_names, input_size, output_size, shared_data, lp_args
        )

    def add_constrs(
        self,
        gvars: list[gurobipy.Var],
        model: gurobipy.Model,
        pre_gvars: list[gurobipy.Var] | None = None,
        pre_bound: ScalarBound | None = None,
    ) -> list[gurobipy.Constr]:
        """
        This is only a placeholder without any constraints.
        """

        pass


class LinearLPNode(BasicLPNode, ABC):
    """
    A class of the linear programming node for linear operation node.

    :param name: The name of the node.
    :param input_names: The names of the input nodes.
    :param input_size: The size of the input data.
    :param output_size: The size of the output data.
    :param shared_data: The shared data between the nodes.
    :param lp_args: The arguments of the linear programming.
    :param weight: Weight of the linear layer.
    :param bias: Bias of the linear layer.
    :param kact_lp_args: The arguments of the k-activation linear programming.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        weight: list[list[float]] | list[list[list[list[float]]]],
        bias: list[float] | None = None,
        kact_lp_args: KActLPArgs | None = None,
    ):
        BasicLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            kact_lp_args,
        )
        self._weight = weight
        self._bias = bias

    @property
    def weight(self) -> list[list[float]] | list[list[list[list[float]]]]:
        return self._weight

    @property
    def bias(self) -> list[float]:
        return self._bias


class GemmLPNode(LinearLPNode):
    """
    A class of the linear programming node for the general matrix multiplication (GEMM).

    :param name: The name of the node.
    :param input_names: The names of the input nodes.
    :param input_size: The size of the input data.
    :param output_size: The size of the output data.
    :param shared_data: The shared data between the nodes.
    :param lp_args: The arguments of the linear programming.
    :param weight: Weight of the linear layer.
    :param bias: Bias of the linear layer.
    :param kact_lp_args: The arguments of the k-activation linear programming.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        weight: list[list[float]],
        bias: list[float] | None = None,
        kact_lp_args: KActLPArgs | None = None,
    ):
        LinearLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            weight,
            bias,
            kact_lp_args,
        )

    def add_constrs(
        self,
        gvars: list[gurobipy.Var],
        model: gurobipy.Model,
        pre_gvars: list[gurobipy.Var] | None = None,
        pre_bound: ScalarBound | None = None,
    ) -> list[gurobipy.Constr]:
        """
        Add gurobi constraints to the gurobi model based on the relaxation for general
        matrix multiplication. This relaxation are linear equations determined by
        the weight and bias of the linear layer.

        :param gvars: Gurobi variables of the current layer.
        :param model: Gurobi model.
        :param pre_gvars: Gurobi variables of the preceding layer.
        :param pre_bound: Scalar bound of the variables of preceding layer.

        :return: A list of gurobi constraints.

        :exception ValueError: If the preceding variables are not provided.
        """
        if pre_gvars is None:
            raise ValueError("pre_gvars are not provided.")

        logger = logging.getLogger("rover")

        constrs = []
        for i in range(len(gvars)):
            wxb = gurobipy.LinExpr(self.weight[i], pre_gvars)
            if self.bias is not None:
                wxb = wxb + self.bias[i]
            y = gvars[i]

            constrs.append(
                model.addLConstr(
                    lhs=wxb,
                    sense=gurobipy.GRB.EQUAL,
                    rhs=y,
                    name=self._create_constr_name(i),
                )
            )

        logger.debug(f"Add {len(constrs)} constraints to the model.")

        return constrs


class Conv2DLPNode(LinearLPNode):
    """
    A class of the linear programming node for the 2D convolutional layer.

    .. attention::
        Currently, we only support the 2D convolutional with some specific parameters.
        The dilation, groups, and ceil mode are not supported.

    :param name: The name of the node.
    :param input_names: The names of the input nodes.
    :param input_size: The size of the input data.
    :param output_size: The size of the output data.
    :param shared_data: The shared data between the nodes.
    :param lp_args: The arguments of the linear programming.
    :param weight: Weight of the convolutional kernel.
    :param bias: Bias of the convolutional kernel.
    :param stride: Stride of the convolutional kernel.
    :param padding: Padding of the convolutional kernel.
    :param dilation: Dilation of the convolutional kernel.
    :param groups: Groups of the convolutional kernel.
    :param ceil_mode: Whether to use the ceil mode.
    :param kact_lp_args: The arguments of the k-activation linear programming.

    :exception ValueError: If the groups is not 1.
    :exception ValueError: If the ceil mode is True.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        output_size: tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        weight: list[list[list[list[float]]]],
        bias: list[float] | None = None,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        ceil_mode: bool = False,
        kact_lp_args: KActLPArgs | None = None,
    ):
        LinearLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            weight,
            bias,
            kact_lp_args,
        )
        if groups != 1:
            raise ValueError(f"groups={groups} is not supported in Conv2DLPNode.")
        if ceil_mode:
            raise ValueError(f"ceil_mode={ceil_mode} is not supported in Conv2DLPNode.")

        self._weight = weight
        self._bias = bias
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._ceil_mode = ceil_mode
        self._kernel_size = (len(weight[0][0]), len(weight[0][0][0]))

    def add_constrs(
        self,
        gvars: list[gurobipy.Var],
        model: gurobipy.Model,
        pre_gvars: list[gurobipy.Var] | None = None,
        pre_bound: ScalarBound = None,
    ) -> list[gurobipy.Constr]:
        """
        Add gurobi constraints to the gurobi model based on the relaxation for 2D
        convolutional layer. This relaxation are linear equations determined by the
        weight and bias of the convolutional layer.

        :param gvars: Gurobi variables of the current layer.
        :param model: Gurobi model.
        :param pre_gvars: Gurobi variables of the preceding layer.
        :param pre_bound: Scalar bound of the variables of preceding layer.

        :return: A list of gurobi constraints.

        :exception ValueError: If the preceding variables are not provided.
        """
        if pre_gvars is None:
            raise ValueError("pre_gvars are not provided.")

        logger = logging.getLogger("rover")

        n_in = int(math.prod(self.input_size))
        stride = self.stride
        dilation = self.dilation
        padding = self.padding

        xcs, xhs, xws = self.input_size
        ycs, yhs, yws = self.output_size
        khs, kws = self.kernel_size
        constrs = []
        for yc, yh, yw in itertools.product(range(ycs), range(yhs), range(yws)):
            y_idx = yc * yhs * yws + yh * yws + yw
            y = gvars[y_idx]
            wxb = gurobipy.LinExpr()

            for xc, kh, kw in itertools.product(range(xcs), range(khs), range(kws)):
                xh = yh * stride[0] + kh * dilation[0] - padding[0]
                if xh < 0 or xh >= xhs:
                    continue
                xw = yw * stride[1] + kw * dilation[0] - padding[1]
                if xw < 0 or xw >= xws:
                    continue
                x_idx = xc * xhs * xws + xh * xws + xw
                if x_idx >= n_in:
                    continue
                wxb += self.weight[yc][xc][kh][kw] * pre_gvars[x_idx]  # noqa

            wxb += self.bias[yc]
            constrs.append(
                model.addLConstr(
                    lhs=wxb,
                    sense=gurobipy.GRB.EQUAL,
                    rhs=y,
                    name=self._create_constr_name(y_idx),
                )
            )

        logger.debug(f"Add {len(constrs)} constraints to the model.")

        return constrs

    @property
    def kernel_size(self) -> tuple:
        """Size of the kernel."""
        return self._kernel_size

    @property
    def stride(self) -> tuple:
        """Stride of the convolutional kernel."""
        return self._stride

    @property
    def dilation(self) -> tuple:
        """Dilation of the convolutional kernel."""
        return self._dilation

    @property
    def padding(self) -> tuple:
        """Padding of the convolutional kernel."""
        return self._padding

    @property
    def groups(self) -> int:
        """Groups of the convolutional kernel."""
        return self._groups

    @property
    def ceil_mode(self) -> bool:
        """Whether to use the ceil mode."""
        return self._ceil_mode


class NonLinearLPNode(BasicLPNode, ABC):
    """
    A class of the linear programming node for the non-linear layer.

    :param name: The name of the node.
    :param input_names: The names of the input nodes.
    :param input_size: The size of the input data.
    :param output_size: The size of the output data.
    :param shared_data: The shared data between the nodes.
    :param lp_args: The arguments of the linear programming.
    :param act_relax_args: The arguments of the activation relaxation.
    :param kact_lp_args: The arguments of the k-activation linear programming.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        BasicLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            kact_lp_args,
        )
        self._act_relax_args = act_relax_args

    def add_constrs(
        self,
        gvars: list[gurobipy.Var],
        model: gurobipy.Model,
        pre_gvars: list[gurobipy.Var] | None = None,
        pre_bound: ScalarBound | None = None,
    ) -> list[gurobipy.Constr]:
        """
        Add gurobi constraints to the gurobi model based on the relaxation for specific
        activation function.

        :param gvars: Gurobi variables of the current layer.
        :param model: Gurobi model.
        :param pre_gvars: Gurobi variables of the preceding layer.
        :param pre_bound: Scalar bound of the variables of preceding layer.

        :return: A list of gurobi constraints.

        :exception ValueError: If the preceding variables are not provided.
        :exception ValueError: If the preceding bounds are not provided.
        """
        if pre_gvars is None:
            raise ValueError("pre_gvars are not provided.")
        if pre_bound is None:
            raise ValueError("pre_bounds are not provided.")

        logger = logging.getLogger("rover")

        pre_lower_bounds = pre_bound.l.flatten().tolist()
        pre_upper_bounds = pre_bound.u.flatten().tolist()
        constrs = []
        for i, (l, u) in enumerate(zip(pre_lower_bounds, pre_upper_bounds)):
            x, y = pre_gvars[i], gvars[i]
            constrs.extend(
                self.cal_single_neuron_relaxation(
                    model, x, y, l, u, name=self._create_var_name(i)
                )
            )

        logger.debug(f"Add {len(constrs)} constraints to the model.")

        return constrs

    @abstractmethod
    def cal_single_neuron_relaxation(
        self,
        model: gurobipy.Model,
        x: gurobipy.Var,
        y: gurobipy.Var,
        l: float,
        u: float,
        name: str,
    ) -> list[gurobipy.Constr]:
        """
        Calculate the relaxation of a single neuron.

        :param model: Gurobi model.
        :param x: Variable of the input.
        :param y: Variable of the output.
        :param l: Lower bound of the input.
        :param u: Upper bound of the input.
        :param name: The name of the constraint.

        :return: list of constraints
        """
        pass

    def add_kact_constrs(
        self,
        pre_gvars: list[gurobipy.Var],
        gvars: list[gurobipy.Var],
        grouped_input_ids: list[list[int]],
        grouped_output_ids: list[list[int]],
        grouped_constrs: np.ndarray | list[np.ndarray],
        model: gurobipy.Model,
    ) -> list[gurobipy.Constr]:
        """
        Add k-activation constraints to the gurobi model.

        :param pre_gvars: Gurobi variables of the preceding layer.
        :param gvars: Gurobi variables of the current layer.
        :param grouped_input_ids: Grouped neuron indices of the input variables,
            i.e., the variables of preceding layer.
        :param grouped_output_ids: Grouped neuron indices of the output variables,
            i.e., the variables of current non-linear layer.
        :param grouped_constrs: Grouped k-activation constraints.
        :param model: Gurobi model.

        :return: A list of Guorbi constraints.
        """

        logger = logging.getLogger("rover")
        k = len(grouped_input_ids[0])
        gconstrs = []

        def get_name(i_, j_):
            return f"{self.__class__.__name__[:-7]}_{i_}_{j_}_kact"

        # For normal unary activation function, the grouped_input_ids and
        # grouped_output_ids are the same.
        num_none = 0
        for i, (ids, constrs) in enumerate(zip(grouped_input_ids, grouped_constrs)):
            if constrs is None:
                num_none += 1
                continue
            for j, coeffs in enumerate(constrs):
                name = get_name(i, j)
                b, c_x, c_y = coeffs[0], coeffs[1 : k + 1], coeffs[k + 1 :]
                x, y = [pre_gvars[i] for i in ids], [gvars[i] for i in ids]
                wx, wy = gurobipy.LinExpr(c_x, x), gurobipy.LinExpr(c_y, y)
                gconstrs.append(
                    model.addLConstr(
                        lhs=b + wx + wy,
                        sense=gurobipy.GRB.GREATER_EQUAL,
                        rhs=0.0,
                        name=name,
                    )
                )

        logger.info(f"{num_none} groups are none due to tiny input polytope.")
        model.update()
        logger.info(
            f"Add {len(gconstrs)} k-activation constraints of "
            f"layer {self.name} to the model."
        )

        return gconstrs

    @staticmethod
    @abstractmethod
    def f(x: float) -> float:
        """
        Activation function.

        :param x: Input value.

        :return: Output value.
        """
        pass

    @staticmethod
    @abstractmethod
    def df(x: float) -> float:
        """
        Derivative of the activation function.

        :param x: Input value.

        :return: Derivative of the activation function.
        """
        pass

    @property
    def act_relax_args(self):
        """Arguments of the activation relaxation."""
        return self._act_relax_args

    @property
    def kact_lp_args(self):
        """Arguments of the k-activation linear programming."""
        return self._kact_lp_args


class ReLULPNode(NonLinearLPNode):
    """
    A class of the linear programming node for the ReLU activation function.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        NonLinearLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            act_relax_args,
            kact_lp_args,
        )

    def cal_single_neuron_relaxation(
        self,
        model: gurobipy.Model,
        x: gurobipy.Var,
        y: gurobipy.Var,
        l: float,
        u: float,
        name: str,
    ) -> list[gurobipy.Constr]:

        if l >= 0:
            constr = model.addLConstr(lhs=x, sense=gurobipy.GRB.EQUAL, rhs=y, name=name)
            return [constr]

        elif l < 0 < u:
            min_half_range = self.act_relax_args.min_half_range
            if u > min_half_range and l < -min_half_range:
                constr1 = model.addLConstr(
                    lhs=y, sense=gurobipy.GRB.GREATER_EQUAL, rhs=x, name=name + "_l_act"
                )
                constr2 = model.addLConstr(
                    lhs=u * (x - l),
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=(u - l) * y,
                    name=name + "_u_act",
                )

            else:
                constr1 = model.addLConstr(
                    lhs=y, sense=gurobipy.GRB.GREATER_EQUAL, rhs=x, name=name + "_l_act"
                )
                constr2 = model.addLConstr(
                    lhs=x - l,
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=y,
                    name=name + "_u_act",
                )
            return [constr1, constr2]

        elif u <= 0:
            return []

        else:
            raise ValueError(f"Invalid bounds: {l}, {u}")

    @staticmethod
    def f(x: float) -> float:
        return relu(x)

    @staticmethod
    def df(x: float) -> float:
        return drelu(x)


class LeakyReLULPNode(NonLinearLPNode):
    """
    A class of the linear programming node for the Leaky ReLU activation function.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        NonLinearLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            act_relax_args,
            kact_lp_args,
        )

    def cal_single_neuron_relaxation(
        self,
        model: gurobipy.Model,
        x: gurobipy.Var,
        y: gurobipy.Var,
        l: float,
        u: float,
        name: str,
    ) -> list[gurobipy.Constr]:

        yl, yu = self.f(l), self.f(u)

        if l >= 0.0:
            constr = model.addLConstr(lhs=y, sense=gurobipy.GRB.EQUAL, rhs=x, name=name)
            return [constr]

        elif l < 0.0 < u:
            if (
                u > self.act_relax_args.min_half_range
                and l < -self.act_relax_args.min_half_range
            ):
                constr11 = model.addLConstr(
                    lhs=y,
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=x,
                    name=name + "_l1_act",
                )
                constr12 = model.addLConstr(
                    lhs=y,
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=0.01 * x,
                    name=name + "_l2_act",
                )
                constr2 = model.addLConstr(
                    lhs=(yu - yl) * (x - l) + (u - l) * yl,
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=(u - l) * y,
                    name=name + "_u_act",
                )
                return [constr11, constr12, constr2]
            else:
                constr1 = model.addLConstr(
                    lhs=y, sense=gurobipy.GRB.GREATER_EQUAL, rhs=x, name=name + "_l_act"
                )
                constr2 = model.addLConstr(
                    lhs=x - l,
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=y,
                    name=name + "_u_act",
                )
                return [constr1, constr2]

        elif u <= 0.0:
            constr = model.addLConstr(
                lhs=y, sense=gurobipy.GRB.EQUAL, rhs=0.01 * x, name=name
            )
            return [constr]

        else:
            raise ValueError(f"Invalid bounds: {l}, {u}")

    @staticmethod
    def f(x: float) -> float:
        return leakyrelu(x)

    @staticmethod
    def df(x: float) -> float:
        return dleakyrelu(x)


class ELULPNode(NonLinearLPNode):
    """
    A class of the linear programming node for the ELU activation function.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        NonLinearLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            act_relax_args,
            kact_lp_args,
        )

    def cal_single_neuron_relaxation(
        self,
        model: gurobipy.Model,
        x: gurobipy.Var,
        y: gurobipy.Var,
        l: float,
        u: float,
        name: str,
    ) -> list[gurobipy.Constr]:

        if l >= 0.0:
            return [model.addLConstr(lhs=y, sense=gurobipy.GRB.EQUAL, rhs=x, name=name)]

        else:
            yl, yu = self.f(l), self.f(u)
            kl, ku = self.df(l), self.df(u)
            k = (yu - yl) / (u - l)
            constr1 = model.addLConstr(
                lhs=k * (x - l) + yl,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=y,
                name=name + "_u_act",
            )
            constr21 = model.addLConstr(
                lhs=y,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=kl * (x - l) + yl,
                name=name + "_l1_act",
            )
            constr22 = model.addLConstr(
                lhs=y,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=ku * (x - u) + yu,
                name=name + "_l2_act",
            )

            if u - l < self.act_relax_args.min_range:
                return [constr1, constr21, constr22]

            m = (u + l) / 2.0
            ym = self.f(m)
            km = self.df(m)
            constr23 = model.addLConstr(
                lhs=y,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=km * (x - m) + ym,
                name=name + "_l3_act",
            )

            return [constr1, constr21, constr22, constr23]

    @staticmethod
    def f(x: float) -> float:
        return elu(x)

    @staticmethod
    def df(x: float) -> float:
        return delu(x)


class SShapeLPNode(NonLinearLPNode, ABC):
    """
    A class of the linear programming node for the S-shaped activation function.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        NonLinearLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            act_relax_args,
            kact_lp_args,
        )

    def cal_single_neuron_relaxation(
        self,
        model: gurobipy.Model,
        x: gurobipy.Var,
        y: gurobipy.Var,
        l: float,
        u: float,
        name: str,
    ):
        if np.allclose(l, u):
            f, df = self.f, self.df
            return [
                model.addLConstr(
                    lhs=y,
                    sense=gurobipy.GRB.EQUAL,
                    rhs=f(l) + df(l) * (x - l),
                    name=name,
                )
            ]

        else:
            if self.__class__ == SigmoidLPNode:
                limit = self.act_relax_args.sigmoid_limit_bound
            else:
                limit = self.act_relax_args.tanh_limit_bound
            if u - l < self.act_relax_args.min_range or u < -limit or l > limit:
                f, df = self.f, self.df
                s = min(df(l), df(u))
                constr1 = model.addLConstr(
                    lhs=y,
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=f(l) + s * (x - l),
                    name=name + "_l_act",
                )
                constr2 = model.addLConstr(
                    lhs=f(u) + s * (x - u),
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=y,
                    name=name + "_u_act",
                )
                return [constr1, constr2]

        cs = self._get_neuron_wise_constraints(l, u)
        constrs = []
        for i, c in enumerate(cs):
            name = name
            b, kx, ky = c
            constr = model.addLConstr(
                lhs=b + kx * x + ky * y,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=0,
                name=name + f"_{i}_act",
            )
            constrs.append(constr)

        return constrs

    def _get_neuron_wise_constraints(self, l: float, u: float) -> list[list[float]]:
        f, df = self.f, self.df
        xl, xu = l, u
        yl, yu = f(xl), f(xu)
        kl, ku = df(xl), df(xu)
        bl, bu = yl - kl * xl, yu - ku * xu
        klu = (yu - yl) / (xu - xl)

        # Triangle relaxation + one more line
        if ku >= klu:
            bu, ku, _ = (
                self._get_second_tangent_line(xu, get_big=False)
                if xu > 0
                else (bu, ku, None)
            )
            blu2, klu2, _ = self._get_parallel_tangent_line(klu, get_big=False)
            blu = yl - klu * xl
            return [
                [blu, klu, -1.0],
                [-bl, -kl, 1.0],
                [-bu, -ku, 1.0],
                [-blu2, -klu2, 1.0],
            ]

        elif kl >= klu:
            btu, ktu, _ = (
                self._get_second_tangent_line(xl, get_big=True)
                if xl < 0
                else (bl, kl, None)
            )
            blu2, klu2, _ = self._get_parallel_tangent_line(klu, get_big=True)
            blu = yl - klu * xl
            return [
                [bl, kl, -1.0],
                [bu, ku, -1.0],
                [blu2, klu2, -1.0],
                [-blu, -klu, 1.0],
            ]

        # Hexagon relaxation
        btu, ktu, _ = self._get_second_tangent_line(xl, get_big=True)
        btl, ktl, _ = self._get_second_tangent_line(xu, get_big=False)
        blul, klul, _ = self._get_parallel_tangent_line(klu, get_big=False)
        bluu, kluu, _ = self._get_parallel_tangent_line(klu, get_big=True)

        return [
            [-bl, -kl, 1.0],
            [-btl, -ktl, 1.0],
            [-blul, -klul, 1.0],
            [bu, ku, -1.0],
            [btu, ktu, -1.0],
            [bluu, kluu, -1.0],
        ]

    @staticmethod
    @abstractmethod
    def _get_second_tangent_line(
        x: float,
        get_big: bool,
    ) -> tuple[float, float, float]:
        pass

    @staticmethod
    @abstractmethod
    def _get_parallel_tangent_line(
        k: float, get_big: bool
    ) -> tuple[float, float, float]:
        pass


class SigmoidLPNode(SShapeLPNode):
    """
    A class of the linear programming node for the sigmoid activation function.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        SShapeLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            act_relax_args,
            kact_lp_args,
        )

    @staticmethod
    def f(x: float) -> float:
        return sigmoid(x)

    @staticmethod
    def df(x: float) -> float:
        return dsigmoid(x)

    @staticmethod
    def _get_second_tangent_line(x: float, get_big: bool) -> tuple[float, float, float]:
        return get_second_tangent_line(x, get_big, "sigmoid")

    @staticmethod
    def _get_parallel_tangent_line(
        k: float, get_big: bool
    ) -> tuple[float, float, float]:
        return get_parallel_tangent_line(k, get_big, "sigmoid")


class TanhLPNode(SShapeLPNode):
    """
    A class of the linear programming node for the tanh activation function.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kact_lp_args: KActLPArgs | None = None,
    ):
        SShapeLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            act_relax_args,
            kact_lp_args,
        )

    @staticmethod
    def f(x: float) -> float:
        return tanh(x)

    @staticmethod
    def df(x: float) -> float:
        return dtanh(x)

    @staticmethod
    def _get_second_tangent_line(x: float, get_big: bool) -> tuple[float, float, float]:
        return get_second_tangent_line(x, get_big, "tanh")

    @staticmethod
    def _get_parallel_tangent_line(
        k: float, get_big: bool
    ) -> tuple[float, float, float]:
        return get_parallel_tangent_line(k, get_big, "tanh")


class MaxPool2DLPNode(NonLinearLPNode):
    """
    A class of the linear programming node for the 2D max pooling layer.

    .. attention::
        Currently, we only support the 2D max pooling with some specific parameters.
        The delay and ceil mode are not supported.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
        act_relax_args: ActRelaxArgs,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        ceil_mode: bool = False,
        kact_lp_args: KActLPArgs | None = None,
    ):
        NonLinearLPNode.__init__(
            self,
            name,
            input_names,
            input_size,
            output_size,
            shared_data,
            lp_args,
            act_relax_args,
            kact_lp_args,
        )

        if ceil_mode:
            raise ValueError(f"ceil_mode={ceil_mode} is not supported in Conv2DLPNode.")

        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._ceil_mode = ceil_mode

    def add_constrs(
        self,
        gvars: list[gurobipy.Var],
        model: gurobipy.Model,
        pre_gvars: list[gurobipy.Var] | None = None,
        pre_bound: ScalarBound = None,
    ) -> list[gurobipy.Constr]:
        """
        Add gurobi constraints to the gurobi model based on the max pooling.

        :param gvars: Gurobi variables of the current layer.
        :param model: Gurobi model.
        :param pre_gvars: Gurobi variables of the preceding layer.
        :param pre_bound: Scalar bound of the variables of preceding layer.

        :return: A list of gurobi constraints.

        :exception ValueError: If the preceding variables are not provided.
        :exception ValueError: If the preceding bounds are not provided.
        """
        logger = logging.getLogger("rover")

        if pre_gvars is None:
            raise ValueError("pre_gvars are not provided.")
        if pre_bound is None:
            raise ValueError("pre_bounds are not provided.")

        constrs = []

        l = pre_bound.l.flatten().tolist()
        u = pre_bound.u.flatten().tolist()

        n_in = int(math.prod(self.input_size))
        stride = self.stride
        dilation = self.dilation
        padding = self.padding

        xcs, xhs, xws = self.input_size
        ycs, yhs, yws = self.output_size
        xks, yks = self.kernel_size
        for yc, yh, yw in itertools.product(range(ycs), range(yhs), range(yws)):
            y_idx = yc * yhs * yws + yh * yws + yw
            y = gvars[y_idx]
            x_sum = gurobipy.LinExpr()
            l_sum = 0.0
            l_max = -float("inf")
            u_max = -float("inf")
            xc = yc

            for xk, yk in itertools.product(range(xks), range(yks)):
                xh = yh * stride[0] + xk * dilation[0] - padding[0]
                if xh < 0 or xh >= xhs:
                    continue
                xw = yw * stride[1] + yk * dilation[0] - padding[1]
                if xw < 0 or xw >= xws:
                    continue
                x_idx = xc * xhs * xws + xh * xws + xw
                if x_idx >= n_in:
                    continue

                name = self._create_constr_name(y_idx) + f"_l_{x_idx}"
                constrs.append(
                    model.addLConstr(
                        lhs=y,
                        sense=gurobipy.GRB.GREATER_EQUAL,
                        rhs=pre_gvars[x_idx],
                        name=name,
                    )
                )
                model.update()

                x_sum += pre_gvars[x_idx]
                l_sum += l[x_idx]
                l_max = max(l_max, l[x_idx])
                u_max = max(u_max, u[x_idx])

            name = self._create_constr_name(y_idx) + "_u"
            constrs.append(
                model.addLConstr(
                    lhs=x_sum - l_sum + l_max,
                    # lhs=u_max,
                    sense=gurobipy.GRB.GREATER_EQUAL,
                    rhs=y,
                    name=name,
                )
            )

        logger.debug(f"Add {len(constrs)} constraints to the model.")

        return constrs

    def add_kact_constrs(
        self,
        pre_gvars: list[gurobipy.Var],
        gvars: list[gurobipy.Var],
        grouped_input_ids: list[list[int]],
        grouped_output_ids: list[list[int]],
        grouped_constrs: np.ndarray | list[np.ndarray],
        model: gurobipy.Model,
    ) -> list[gurobipy.Constr]:
        """
        Add k-activation constraints to the gurobi model.

        .. tip::
            We calculate the function hull of a max pool, so all the inputs of the pool
            is the input variables but the output variable is only one.
            In this case, the inputs and outputs are not one-to-one mapping.
            Because of this, we need the output ids to get the output variable.

            For other activation functions, the function hull has multiple input
            dimensions and multiple output dimensions by grouping strategy, where we
            group multiple neurons into one group and calculate the function hull of
            the group.

        :param pre_gvars: Gurobi variables of the preceding layer.
        :param gvars: Gurobi variables of the current layer.
        :param grouped_input_ids: Grouped neuron indices of the input variables,
            i.e., the variables of preceding layer.
        :param grouped_output_ids: Grouped neuron indices of the output variables,
            i.e., the variables of current non-linear layer.
        :param grouped_constrs: Grouped k-activation constraints.
        :param model: Gurobi model.

        :return: A list of Guorbi constraints.
        """

        logger = logging.getLogger("rover")
        k = len(grouped_input_ids[0])
        gconstrs = []

        def get_name(i_, j_):
            return f"{self.__class__.__name__[:-7]}_{i_}_{j_}_kact"

        num_none = 0
        for i, (input_ids, output_id, constrs) in enumerate(
            zip(grouped_input_ids, grouped_output_ids, grouped_constrs)
        ):
            if constrs is None:
                num_none += 1
                continue
            for j, coeffs in enumerate(constrs):
                name = get_name(i, j)
                b, c_x, c_y = coeffs[0], coeffs[1 : k + 1], coeffs[k + 1 :]
                x, y = [pre_gvars[i] for i in input_ids], gvars[output_id]  # noqa
                wx, wy = gurobipy.LinExpr(c_x, x), gurobipy.LinExpr(c_y, y)
                gconstrs.append(
                    model.addLConstr(
                        lhs=b + wx + wy,
                        sense=gurobipy.GRB.GREATER_EQUAL,
                        rhs=0.0,
                        name=name,
                    )
                )
        logger.info(f"{num_none} groups are none due to tiny input polytope.")
        model.update()
        logger.info(
            f"Add {len(gconstrs)} k-activation constraints of "
            f"{self.name} to the model."
        )

        return gconstrs

    def cal_single_neuron_relaxation(
        self,
        model: gurobipy.Model,
        x: gurobipy.Var,
        y: gurobipy.Var,
        l: float,
        u: float,
        name: str,
    ) -> list[gurobipy.Constr]:
        raise NotSupported()

    @staticmethod
    def df(x: float) -> float:
        raise NotSupported()

    @staticmethod
    def f(x: float) -> float:
        raise NotSupported()

    @property
    def kernel_size(self) -> tuple:
        """Kernel size of the max pooling."""
        return self._kernel_size

    @property
    def stride(self) -> tuple:
        """Stride of the max pooling."""
        return self._stride

    @property
    def padding(self) -> tuple:
        """Padding of the max pooling."""
        return self._padding

    @property
    def dilation(self) -> tuple:
        """Diolation of the max pooling."""
        return self._dilation

    @property
    def ceil_mode(self) -> bool:
        """Whether to use the ceil mode."""
        return self._ceil_mode


class ResidualAddLPNode(BasicLPNode):
    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        output_size: tuple[int] | tuple[int, int, int],
        shared_data: LPSharedData,
        lp_args: LPArgs,
    ):
        BasicLPNode.__init__(
            self, name, input_names, input_size, output_size, shared_data, lp_args
        )

    def add_constrs(
        self,
        gvars: list[gurobipy.Var],
        model: gurobipy.Model,
        pre_gvars1: list[gurobipy.Var] = None,
        pre_bound1: ScalarBound = None,
        pre_gvars2: list[gurobipy.Var] = None,
        pre_bound2: ScalarBound = None,
    ) -> list[gurobipy.Constr]:
        """
        Add gurobi constraints to the gurobi model based on residual addition.

        :param gvars: Gurobi variables of the current layer.
        :param model: Gurobi model.
        :param pre_gvars1: Gurobi variables of the preceding layer 1.
        :param pre_bound1: Scalar bound of the variables of preceding layer 1.
        :param pre_gvars2: Gurobi variables of the preceding layer 2.
        :param pre_bound2: Scalar bound of the variables of preceding layer 2.

        :return: A list of gurobi constraints.

        :exception ValueError: If the preceding variables are not provided.
        :exception ValueError: If the preceding bounds are not provided.
        """
        if pre_gvars1 is None:
            raise ValueError("pre_gvars1 are not provided.")
        if pre_bound1 is None:
            raise ValueError("pre_bounds1 are not provided.")
        if pre_gvars2 is None:
            raise ValueError("pre_gvars2 are not provided.")
        if pre_bound2 is None:
            raise ValueError("pre_bounds2 are not provided.")

        constrs = [
            model.addLConstr(
                lhs=pre_gvars1[i] + pre_gvars2[i],
                sense=gurobipy.GRB.EQUAL,
                rhs=gvars[i],
                name=self._create_constr_name(i),
            )
            for i in range(len(gvars))
        ]

        return constrs
