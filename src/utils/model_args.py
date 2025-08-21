"""
This module provides arguments for different approaches.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "LoggerArgs",
    "PerturbationArgs",
    "ActRelaxArgs",
    "LPArgs",
    "KActLPArgs",
]

import logging
import os
from dataclasses import dataclass
from torch import Tensor

from .enums import *


@dataclass
class LoggerArgs:
    """
    Arguments for the logger.
    """

    log_level: int = logging.INFO
    """
    Logging level. The default value is `logging.INFO`.
    
    ..seealso::
        Refer to the `logging` for the details.
    """

    log_file: str | None = None
    """
    The file path to save the log. If it is None, the log will be printed to the 
    console.
    """

    log_format: str = (
        "%(asctime)s "
        "[%(levelname)s] "
        "%(message)-100s - "
        "%(pathname)s, "
        # "%(name)s - "
        # "%(module)s - "
        # "%(funcName)s - "
        "Line: %(lineno)d"
    )

    """The format of the log."""

    log_console: bool = False
    """Whether to print the log to the console."""

    def __post_init__(self):
        if self.log_file:
            # Create the dir
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def __str__(self):
        return (
            f"LoggerArgs("
            f"log_level={self.log_level}, "
            f"log_file={self.log_file}, "
            f"log_format={self.log_format}, "
            f"log_console={self.log_console}"
            f")"
        )


@dataclass
class PerturbationArgs:
    """
    Arguments for the perturbation.

    :exception ValueError: If the `epsilon` is negative.
    :exception ValueError: If the `norm` is not of `float`.
    """

    epsilon: float
    """The perturbation radius."""

    norm: float = float("inf")
    """
    The norm of the perturbation.
    
    .. attention::
        Currently, only infinity norm is supported.
    """

    means: Tensor = Tensor([0.0])
    """The means of the input data."""

    stds: Tensor = Tensor([1.0])
    """The standard deviations of the input data."""

    lower_limit: float = -float("inf")
    """The lower limit of the input data."""

    upper_limit: float = float("inf")
    """The upper limit of the input data."""

    def __post_init__(self):
        if self.epsilon < 0:
            raise ValueError(f"epsilon should be non-negative, got {self.epsilon}")

        if self.norm != float("inf"):
            raise ValueError(f"currently only support infinity norm, got {self.norm}")

    def normalize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Normalize the input data.
        """
        self.means = self.means.to(device=x.device)
        self.stds = self.stds.to(device=x.device)

        l = (
            x.detach()
            .clone()
            .sub_(self.epsilon)
            .clamp_min_(self.lower_limit)
            .sub_(self.means)
            .div_(self.stds)
        )
        u = (
            x.detach()
            .clone()
            .add_(self.epsilon)
            .clamp_max_(self.upper_limit)
            .sub_(self.means)
            .div_(self.stds)
        )

        return l, u

    def __str__(self):
        return (
            f"PerturbationArgs("
            f"epsilon={self.epsilon}, "
            f"means={self.means.tolist()}, "
            f"stds={self.stds.tolist()}, "
            f"lower_limit={self.lower_limit}, "
            f"upper_limit={self.upper_limit})"
        )


@dataclass
class ActRelaxArgs:
    """
    Arguments for the activation relaxation.

    :exception ValueError: If the `mode` is not of `ActRelaxationMode`.
    :exception ValueError: If the `min_range` is not positive.
    :exception ValueError: If the `min_half_range` is not positive.
    :exception ValueError: If the `sigmoid_limit_bound` is not positive.
    :exception ValueError: If the `tanh_limit_bound` is not positive.
    """

    mode: ActRelaxationMode = ActRelaxationMode.ROVER_SN
    """
    The mode of the activation relaxation. The default value is 
    `ActRelaxationMode.ROVER_SN`.
    
    ..seealso::
        Refer to the `ActRelaxationMode` for the details
    """

    min_range: float = 0.1
    """The minimum input range to calculate a complicated linear relaxation. The 
    default value is 0.1.
    
    For ReLU, this number can be small.
    For sigmoid and tanh, this number should be large enough to avoid numerical issues.
    
    .. attention::
        The value should not be too small, otherwise, numerical issues may occur.
        This value does not affect the bound propagation relaxation.
        This value directly affects the linear programming relaxation.
        The function hull relaxation will consider further specifically conditions.
    """

    min_half_range: float = min_range / 2
    """
    The minimum half range to calculate a complicated linear relaxation. The default
    value is `min_range / 2`, which reuiqres the lower bound should smaller than
    -min_half_range and the upper bound should greater than min_half_range.
    
    .. attention::
        The value should not be too small, otherwise, numerical issues may occur.
        This value does not affect the bound propagation relaxation.
        This value directly affects the linear programming relaxation.
        The function hull relaxation will consider further specifically conditions.
    """

    sigmoid_limit_bound: float = 5.0
    """
    The limit bound for the sigmoid function. The default value is 5.0.
    
    .. hint::
        When the input is greater than the limit bound, the sigmoid function will be
        approximated as 1.0. When the input is smaller than the negative limit bound,
        the sigmoid function will be approximated as 0.0. So we will use simpler linear
        relaxation for these cases.
        
    .. attention::
        The value should not be too big, otherwise, numerical issues may occur.
        This value does not affect the bound propagation relaxation.
        This value directly affects the linear programming relaxation.
        The function hull relaxation will consider further specifically conditions.
    """

    tanh_limit_bound: float = 3.0
    """
    The limit bound for the tanh function. The default value is 3.0.
    
    .. hint::
        When the input is greater than the limit bound, the tanh function will be
        approximated as 1.0. When the input is smaller than the negative limit bound,
        the tanh function will be approximated as -1.0. So we will use simpler linear
        relaxation for these cases.
        
    .. attention::
        The value should not be too big, otherwise, numerical issues may occur.
        This value does not affect the bound propagation relaxation.
        This value directly affects the linear programming relaxation.
        The function hull relaxation will consider further specifically conditions.
    """

    update_scalar_bounds_per_layer: bool = True
    """
    Whether to update the scalar bounds for each back-substitution of each layer.
    
    .. hint::
        There will be an improvement for MaxPool layer following the ReLU layer. Two 
        adjacent non-linear layers needs this to update the scalar bounds by the output
        range of the activation function.
        So we calculate bounds for all layer and update the scalar bounds for each layer
        in each back-substitution.
    """

    def __post_init__(self):
        if not isinstance(self.mode, ActRelaxationMode):
            raise ValueError(f"mode should be of ActRelaxationMode, got {self.mode}")
        if self.min_range < 0:
            raise ValueError(f"min_range should be nono-negative, got {self.min_range}")
        if self.min_half_range < 0:
            raise ValueError(
                f"min_half_range should be non-negative, got {self.min_half_range}"
            )
        if self.sigmoid_limit_bound <= 0:
            raise ValueError(
                f"sigmoid_limit_bound should be positive, got "
                f"{self.sigmoid_limit_bound}"
            )
        if self.tanh_limit_bound <= 0:
            raise ValueError(
                f"tanh_limit_bound should be positive, got {self.tanh_limit_bound}"
            )

    def __str__(self):
        return (
            f"ActRelaxArgs("
            f"mode={self.mode}, "
            f"min_range={self.min_range}, "
            f"min_half_range={self.min_half_range}, "
            f"sigmoid_limit_bound={self.sigmoid_limit_bound}, "
            f"tanh_limit_bound={self.tanh_limit_bound})"
        )


@dataclass
class LPArgs:
    """
    Arguments for the linear programming relaxation.

    We use Gurobi arguments to control the optimization process.

    .. seealso::
        Refer to the

        - `Gurobi parameters
          <https://www.gurobi.com/documentation/current/refman/parameters.html>`_.
        - `Making the algorithm less sensitive
          <https://www.gurobi.com/documentation/current/refman
          /making_the_algorithm_less_.html>`_.
        - `Method selection
          <https://www.gurobi.com/documentation/current/refman/method.html>`_.
    """

    terminate_if_fail: bool = True
    """
    Whether to terminate the optimization when the verification of one label fails.
    
    .. hint::
    
        For classification tasks, we need to verify each label.
        Specifically, we verify the lower bound of the target label is greater than the
        upper bound of the other labels, which is to verify the difference between the
        target label and the other labels is greater than 0.
    """

    gurobi_output_flag: int = 0
    """The output flag for the Gurobi solver. The default value is 0."""

    gurobi_timelimit: int = 60
    """The time limit for the Gurobi solver. The default value is 60."""

    gurobi_numeric_focus: int = 0
    """The numeric focus for the Gurobi solver. The default value is 2."""

    gurobi_optimality_tol: float = 1e-9
    """The optimality tolerance for the Gurobi solver. The default value is 1e-9."""

    gurobi_feasibility_tol: float = 1e-4
    """The feasibility tolerance for the Gurobi solver. The default value is 1e-4."""

    gurobi_markowitz_tol: float = 0.99
    """The Markowitz tolerance for the Gurobi solver. The default value is 0.99."""

    gurobi_method: int = 3
    """
    The method for the Gurobi solver. The default value is 3, which is to use
    different methods (simplex method, interior point method, and dual method) 
    parallelly.
    Refer to the `Method selection 
    <https://www.gurobi.com/documentation/current/refman/method.html>`_. for the
    details.
    """

    gurobi_cutoff: float = 1e-6
    """The cutoff for the Gurobi solver. The default value is 1e-6. This is to stop 
    the optimization when we can infer that the objective value is less than the 
    cutoff."""

    def __str__(self):
        return (
            f"LPArgs("
            f"gurobi_output_flag={self.gurobi_output_flag}, "
            f"gurobi_timelimit={self.gurobi_timelimit}, "
            f"gurobi_numeric_focus={self.gurobi_numeric_focus}, "
            f"gurobi_optimiality_tol={self.gurobi_optimality_tol}, "
            f"gurobi_feasibility_tol={self.gurobi_feasibility_tol}, "
            f"gurobi_markowitz_tol={self.gurobi_markowitz_tol}, "
            f"gurobi_method={self.gurobi_method}, "
            f"gurobi_cutoff={self.gurobi_cutoff})"
        )


@dataclass
class KActLPArgs:
    """
    Arguments for the KAct relaxation.

    We use :class:`ActHull` to calculate the multi-neuron constriants for
    linear programming.

    :exception ValueError: If the `partition_size` is not positive.
    :exception ValueError: If the `group_size` is not positive.
    :exception ValueError: If the `max_overlap_size` is negative.
    :exception ValueError: If the `max_parallel_groups` is not positive.
    :exception ValueError: If the `min_range` is negative.
    :exception ValueError: If the `min_limit` is negative.
    :exception ValueError: If the `max_groups` is not positive.
    :exception ValueError: If the `gurobi_lazy_callback_objval` is negative.

    .. seealso::
        This is inspired by the paper

        - `Beyond the single neuron convex barrier for neural network certification
          <https://proceedings.neurips.cc/paper_files/paper/2019/file
          /0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf>`_
          and
        - `PRIMA: general and precise neural network certification via scalable
          convex hull approximations <https://dl.acm.org/doi/pdf/10.1145/3498704>`_.
    """

    partition_size: int = 50
    """
    The partition size is to partition the neurons in separate groups. The default
    value is 50.
    """

    group_size: int = 4
    """
    The group size is to group the neurons in the same group. The default value is 4.
    
    .. hint::
    
      The gouping strategy is to first partition the neurons into separate groups and
      then group the neurons in the same partion. The neurons in the same group is to
      calculate the function hull.
    """

    max_overlap_size: int = 1
    """
    The maximum overlap size is to control the overlap size between the groups. The
    default value is 1.
    """

    max_parallel_groups: int = 500
    """
    The maximum number of parallel groups calculation for the KAct relaxation. 
    The default value is 100.
    """

    max_groups: int = 2000
    """
    The maximum number of groups to calculate for the KAct relaxation.
    If the value is -1, all the groups will be calculated.
    """

    use_multi_threads: bool = True
    """
    Whether to use multi-threads for the KAct relaxation.
    The default value is False.
    """

    use_lazy_constraints: bool = True
    """
    Whether to use lazy constraints for the KAct relaxation.
    
    .. hint::
        This will selectively add the multi-neuron constraints to improve the objective
        value after solve a relatively simple LP problem without multi-neuron 
        constraints.
        The default value is False.
    """

    gurobi_lazy_callback_objval: float = -5.0
    """
    The objective value for the lazy callback for the Gurobi solver.
    
    .. hint::
        If the objective value is less than this value, the lazy constraints will not be
        further added to improve the objective value.
    """

    constr_template: ConstrTemplate = ConstrTemplate.HEXAGON
    """
    The template for the constraints. 
    The default value is `ConstrTemplate.HEXAGON`.
    Refer to the `ConstrTemplate` for the details.
    """

    def __post_init__(self):
        if self.partition_size <= 0:
            raise ValueError(
                f"partition_size should be positive, got {self.partition_size}"
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size should be positive, got {self.group_size}")
        if self.max_overlap_size < 0:
            raise ValueError(
                f"max_overlap_size should be non-negative, got {self.max_overlap_size}"
            )
        if self.max_parallel_groups <= 0:
            raise ValueError(
                f"max_parallel_groups should be positive, got "
                f"{self.max_parallel_groups}"
            )
        if self.max_groups < -1:
            raise ValueError(f"max_groups should be >= -1 got {self.max_groups}")
        if self.gurobi_lazy_callback_objval > 0:
            raise ValueError(
                f"gurobi_lazy_callback_objval should be non-positive, "
                f"got {self.gurobi_lazy_callback_objval}"
            )

    def __str__(self):
        return (
            f"KActArgs("
            f"partition_size={self.partition_size}, "
            f"group_size={self.group_size}, "
            f"max_overlap_size={self.max_overlap_size}, "
            f"max_parallel_groups={self.max_parallel_groups}, "
            f"max_groups={self.max_groups}, "
            f"use_multi_threads={self.use_multi_threads}, "
            f"use_lazy_constraints={self.use_lazy_constraints}, "
            f"gurobi_lazy_callback_objval={self.gurobi_lazy_callback_objval}, "
            f"constr_template={self.constr_template})"
        )
