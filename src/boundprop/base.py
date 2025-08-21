"""
This module defines the base class for all the nodes in the bound propagation.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "BasicNode",
    "InputNode",
    "LinearNode",
    "GemmNode",
    "Conv2DNode",
    "NonLinearNode",
    "ReLUNode",
    "SigmoidNode",
    "TanhNode",
    "ELUNode",
    "LeakyReLUNode",
    "MaxPool2DNode",
    "ResidualAddNode",
]
from abc import abstractmethod, ABC
from typing import TypeVar, Generic

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .containers import BPSharedData
from .nontrivial_neuron import *
from .utils import *
from ..utils import *

T = TypeVar("T", bound="BasicNode")


class BasicNode(Module, ABC, Generic[T]):
    """
    An abstract class for the basic node.

    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
    ):
        Module.__init__(self)

        self._name = name
        self._input_names = input_names

        # All input/output sizes are the data size without the batch size.
        self._input_size = input_size

        self.shared_data = shared_data

        # If the pre_nodes is None, it means that the node is the input node.
        self._pre_nodes: list[T] | None = None

        # If the next_nodes is None, it means that the node is the output node.
        self._next_nodes: list[T] | None = None

        # This is calculated in the child class.
        self._output_size: tuple[int] | tuple[int, int, int] | None = None

        # This is for typing check.
        self._ni: int = math.prod(input_size)
        self._no: int

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The forward function of the node aims to calculate the bound or relaxation of
        the corresponding layer.
        It should be implemented in the child class.
        """
        pass

    @abstractmethod
    def _cal_output_size(self) -> tuple[int] | tuple[int, int, int]:
        """
        Calculate the output size of the layer based on the input size and other
        parameters like kernel size, stride, padding, and dilation.

        :return: The output size of the layer.
        """
        pass

    def clear(self):
        """
        Clear the cache data in the node.
        """
        pass

    @staticmethod
    @abstractmethod
    def cal_bounds(*args, **kwargs) -> ScalarBound:
        """
        Calculate the scalar bound of the node.
        It should be implemented in the child class.
        """
        pass

    @property
    def name(self) -> str:
        """The name of the node."""
        return self._name

    @property
    def input_names(self) -> list[str]:
        """The input names of the node."""
        return self._input_names

    @input_names.setter
    def input_names(self, names: list[str]):
        self._input_names = names

    @property
    def input_size(self) -> tuple[int] | tuple[int, int, int]:
        """The input size of the node."""
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int] | tuple[int, int, int]):
        self._input_size = size
        self._output_size = self._cal_output_size()

    @property
    def output_size(self) -> tuple[int] | tuple[int, int, int]:
        """The output size of the node."""
        return self._output_size

    @output_size.setter
    def output_size(self, size: tuple[int] | tuple[int, int, int]):
        self._output_size = size

    @property
    def pre_nodes(self) -> list[T] | None:
        """The preceding nodes of the node, which have the input names."""
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, nodes: list[T] | None):
        self._pre_nodes = nodes

    @property
    def next_nodes(self) -> list[T] | None:
        """The next nodes of the node, which take the current node as the pre node."""
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, node: list[T] | None):
        self._next_nodes = node

    def __str__(self) -> str:
        return f"{self.__class__.__name__}" f"(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()


class InputNode(BasicNode, ABC):
    """
    An abstract class for the input node.

    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str] | None,
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
    ):
        BasicNode.__init__(self, name, input_names, input_size, shared_data)
        self._output_size = self._cal_output_size()
        self._no = math.prod(self._output_size)

    def _cal_output_size(self) -> tuple[int] | tuple[int, int, int]:
        return self._input_size


class LinearNode(BasicNode, ABC):
    """
    An abstract class for the linear node.

    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        weight: Tensor,
        bias: Tensor | None = None,
    ):
        BasicNode.__init__(self, name, input_names, input_size, shared_data)

        self._weight: Tensor
        self._bias: Tensor | None
        self.register_buffer("_weight", weight)
        if bias is not None:
            self.register_buffer("_bias", bias)
        else:
            self._bias = None

    @property
    def weight(self) -> Tensor:
        """The weight parameter of the node."""
        return self._weight

    @property
    def bias(self) -> Tensor | None:
        """The bias parameter of the node."""
        return self._bias

    @weight.setter
    def weight(self, value: Tensor):
        self._weight = value  # noqa

    @bias.setter
    def bias(self, value: Tensor):
        if self._bias is not None:
            self.register_buffer("_bias", value)
        else:
            self._bias = value


class GemmNode(LinearNode, ABC):
    """
    An abstract class for the gemm (general matrix multiplication) node.

    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation.
    :param weight: The weight of the node.
    :param bias: The bias of the node.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int],
        shared_data: BPSharedData,
        weight: Tensor,
        bias: Tensor | None = None,
    ):
        LinearNode.__init__(
            self,
            name,
            input_names,
            input_size,
            shared_data,
            weight,
            bias,
        )

        self._output_size = self._cal_output_size

        self._no = math.prod(self._output_size)

    @property
    def _cal_output_size(self) -> tuple[int]:
        return (self.weight.size(0),)


class Conv2DNode(LinearNode, ABC):
    """
    An abstract class for the 2D convolution node.

    .. attention::

        We only support the 2D convolution now.

    .. seealso::

        Refer to
        `torch.nn.conv2d()
        <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_
        in PyTorch for more arguments' information.


    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation.
    :param weight: The weight of the convolution kernel.
    :param bias: The bias of the convolution kernel.
    :param stride: The stride of the convolution.
    :param padding: The padding of the convolution.
    :param dilation: The dilation of the convolution.
    :param groups: The number of groups for the convolution.
    :param ceil_mode: The ceil mode for the convolution.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        shared_data: BPSharedData,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        ceil_mode: bool = False,
    ):
        LinearNode.__init__(
            self,
            name,
            input_names,
            input_size,
            shared_data,
            weight,
            bias,
        )

        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._ceil_mode = ceil_mode
        self._output_padding: tuple | None = None

        self._kernel_size: tuple = (self.weight.size(-2), self.weight.size(-1))
        self._output_size = self._cal_output_size()

        self._ci, self._hi, self._wi = self.input_size
        self._co, self._ho, self._wo = self.output_size
        self._no = math.prod(self._output_size)
        self._nk = math.prod(self.output_size[1:])  # number of kernels
        self._nks = math.prod(self.kernel_size)  # number of kernel elements

    def _cal_output_size(self) -> tuple[int, int, int]:

        channel = self.weight.size(0)
        height, width, self._output_padding = cal_conv_data_size(
            self.input_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )

        return channel, height, width

    def clear(self):
        pass

    @property
    def kernel_size(self) -> tuple:
        """The kernel size of the convolution."""
        return self._kernel_size

    @property
    def stride(self) -> tuple:
        """The stride of the convolution."""
        return self._stride

    @property
    def padding(self) -> tuple:
        """The padding of the convolution."""
        return self._padding

    @property
    def output_padding(self) -> tuple:
        """
        The output padding of the convolution.

        .. tip::
            This is used for the transposed convolution, which calculates the
            back-substitution of the convolution.
        """
        if self._output_padding is None:
            self._cal_output_size()

        return self._output_padding

    @property
    def dilation(self) -> tuple:
        """The dilation of the convolution."""
        return self._dilation

    @property
    def groups(self) -> int:
        """The number of groups for the convolution."""
        return self._groups

    @property
    def ceil_mode(self) -> bool:
        """The ceil mode for the convolution."""
        return self._ceil_mode


class NonLinearNode(BasicNode, ABC):
    """
    An abstract class for the non-linear node.

    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation.
    :param act_relax_args: The arguments for the activation relaxation
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        BasicNode.__init__(self, name, input_names, input_size, shared_data)

        self.act_relax_args = act_relax_args
        self.act_type: ActivationType

        self._cached_nontrivial_mask: Tensor | None = None

    def _cal_output_size(self, **kwargs) -> tuple[int] | tuple[int, int, int]:
        return self.input_size

    @abstractmethod
    def cal_relaxation(self, *args, **kwargs):
        """
        Calculate the relaxation of the activation function.
        """
        pass

    def get_nontrivial_neuron_mask(
        self,
        pre_bound: ScalarBound,
        recalculate: bool = False,
        cached: bool = True,
        ignore_degenerate_pool: bool = False,
    ) -> Tensor:
        """
        Get the mask of the neurons that will be handled by nontrivial relaxation.

        :param pre_bound: The scalar bound of the neurons of the preceding layer.
        :param recalculate: Whether to recalculate the mask.
        :param cached: Whether to cache the mask.
        :param ignore_degenerate_pool: Whether to ignore the degenerate neurons,
            which is used for MaxPool2DNode. Refer to the MaxPool2DNode for more
            information.

        :return: The mask of the neurons that will be handled by non-trivial relaxation.
        """
        if self._cached_nontrivial_mask is not None and not recalculate:
            return self._cached_nontrivial_mask

        pre_l, pre_u = pre_bound.l.flatten(), pre_bound.u.flatten()
        mask = self._get_nontrivial_neuron_mask(
            pre_l, pre_u, self.f(pre_l), self.f(pre_u), self.act_relax_args
        ).flatten()

        if cached:
            # Make sure the cached mask has no ignored nontrivial neurons.
            self._cached_nontrivial_mask = mask
            self.shared_data.all_nontrivial_neuron_masks[self.name] = mask

        return mask

    @staticmethod
    @abstractmethod
    def _get_nontrivial_neuron_mask(
        pre_l: Tensor,
        pre_u: Tensor,
        l: Tensor,
        u: Tensor,
        act_relax_args: ActRelaxArgs,
    ) -> Tensor:
        pass

    def clear(self):
        self._cached_nontrivial_mask = None

    @staticmethod
    @abstractmethod
    def f(x: Tensor) -> Tensor:
        """
        The activation function.

        :param x: The input.

        :return: The output.
        """
        pass

    @staticmethod
    @abstractmethod
    def df(x: Tensor) -> Tensor:
        """
        The derivative of the activation function.

        :param x: The input.

        :return: The output.
        """
        pass


class ReLUNode(NonLinearNode, ABC):
    """
    An abstract class for the ReLU node.

    .. tip::
        A ReLU function is defined as

        .. math::
            f(x) = \\max(0, x).
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        NonLinearNode.__init__(
            self,
            name,
            input_names,
            input_size,
            shared_data,
            act_relax_args,
        )
        self._output_size = self._cal_output_size()
        self._no = math.prod(self._output_size)
        self.act_type = ActivationType.RELU

    @staticmethod
    def _get_nontrivial_neuron_mask(
        pre_l: Tensor,
        pre_u: Tensor,
        l: Tensor,
        u: Tensor,
        act_relax_args: ActRelaxArgs,
    ) -> Tensor:
        return get_nontrivial_neuron_mask_relu(pre_l, pre_u, l, u, act_relax_args)

    @staticmethod
    def clamp_bounds(bound: ScalarBound) -> ScalarBound:
        return ScalarBound(torch.clamp_min(bound.l, 0), torch.clamp_min(bound.u, 0))

    @staticmethod
    def f(x: Tensor) -> Tensor:
        return relu(x)

    @staticmethod
    def df(x: Tensor) -> Tensor:
        return drelu(x)


class SigmoidNode(NonLinearNode, ABC):
    """
    An abstract class for the sigmoid node.

    .. tip::
        A sigmoid function is defined as

        .. math::

            f(x) = \\frac{1}{1 + \\exp(-x)}.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        NonLinearNode.__init__(
            self, name, input_names, input_size, shared_data, act_relax_args
        )
        self._output_size = self._cal_output_size()
        self._no = math.prod(self._output_size)
        self.act_type = ActivationType.SIGMOID

    @staticmethod
    def _get_nontrivial_neuron_mask(
        pre_l: Tensor,
        pre_u: Tensor,
        l: Tensor,
        u: Tensor,
        act_relax_args: ActRelaxArgs,
    ) -> Tensor:
        return get_nontrivial_neuron_mask_sigmoid(pre_l, pre_u, l, u, act_relax_args)

    @staticmethod
    def clamp_bounds(bound: ScalarBound) -> ScalarBound:
        return ScalarBound(torch.clamp(bound.l, 0, 1), torch.clamp(bound.u, 0, 1))

    @staticmethod
    def f(x: Tensor) -> Tensor:
        return sigmoid(x)

    @staticmethod
    def df(x: Tensor) -> Tensor:
        return dsigmoid(x)


class TanhNode(NonLinearNode, ABC):
    """
    An abstract class for the tanh node.

    .. tip::
        A tanh function is defined as

        .. math::
            f(x) = \\tanh(x).
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        NonLinearNode.__init__(
            self, name, input_names, input_size, shared_data, act_relax_args
        )
        self._output_size = self._cal_output_size()
        self._no = math.prod(self._output_size)
        self.act_type = ActivationType.TANH

    @staticmethod
    def _get_nontrivial_neuron_mask(
        pre_l: Tensor,
        pre_u: Tensor,
        l: Tensor,
        u: Tensor,
        act_relax_args: ActRelaxArgs,
    ) -> Tensor:
        return get_nontrivial_neuron_mask_relu(pre_l, pre_u, l, u, act_relax_args)

    @staticmethod
    def clamp_bounds(bound: ScalarBound) -> ScalarBound:
        return ScalarBound(torch.clamp(bound.l, -1, 1), torch.clamp(bound.u, -1, 1))

    @staticmethod
    def f(x: Tensor) -> Tensor:
        return tanh(x)

    @staticmethod
    def df(x: Tensor) -> Tensor:
        return dtanh(x)


class ELUNode(NonLinearNode, ABC):
    """
    An abstract class for the ELU node.

    .. tip::
        An ELU function is defined as

        .. math::

            f(x) =
            \\begin{cases}
                x & \\text{if } x > 0, \\\\
                \\alpha (\\exp(x) - 1) & \\text{otherwise},
            \\end{cases}

        where :math:`\\alpha` is a hyperparameter.
        We only consider the case where :math:`\\alpha = 1`.

    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        NonLinearNode.__init__(
            self, name, input_names, input_size, shared_data, act_relax_args
        )
        self._output_size = self._cal_output_size()
        self._no = math.prod(self._output_size)
        self.act_type = ActivationType.ELU

    @staticmethod
    def _get_nontrivial_neuron_mask(
        pre_l: Tensor,
        pre_u: Tensor,
        l: Tensor,
        u: Tensor,
        act_relax_args: ActRelaxArgs,
    ) -> Tensor:
        return get_nontrivial_neuron_mask_elu(pre_l, pre_u, l, u, act_relax_args)

    @staticmethod
    def clamp_bounds(bound: ScalarBound) -> ScalarBound:
        return ScalarBound(torch.clamp_min(bound.l, -1), torch.clamp_min(bound.u, -1))

    @staticmethod
    def f(x: Tensor) -> Tensor:
        return elu(x)

    @staticmethod
    def df(x: Tensor) -> Tensor:
        return delu(x)


class LeakyReLUNode(NonLinearNode, ABC):
    """
    An abstract class for the leaky ReLU node.

    .. tip::
        A leaky ReLU function is defined as

        .. math::
            f(x) =
            \\begin{cases}
                x & \\text{if } x > 0, \\\\
                \\alpha x & \\text{otherwise},
            \\end{cases}

        where :math:`\\alpha` is a hyperparameter.
        We only consider the case where :math:`\\alpha = 0.01`.

    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        NonLinearNode.__init__(
            self, name, input_names, input_size, shared_data, act_relax_args
        )
        self._output_size = self._cal_output_size()
        self._no = math.prod(self._output_size)
        self.act_type = ActivationType.LEAKY_RELU

    @staticmethod
    def _get_nontrivial_neuron_mask(
        pre_l: Tensor,
        pre_u: Tensor,
        l: Tensor,
        u: Tensor,
        act_relax_args: ActRelaxArgs,
    ) -> Tensor:
        return get_nontrivial_neuron_mask_leakyrelu(pre_l, pre_u, l, u, act_relax_args)

    @staticmethod
    def clamp_bounds(bound: ScalarBound):
        return bound

    @staticmethod
    def f(x: Tensor, negative_slope: float = 0.01) -> Tensor:
        return leakyrelu(x)

    @staticmethod
    def df(x: Tensor, negative_slope: float = 0.01) -> Tensor:
        return dleakyrelu(x)


class MaxPool2DNode(NonLinearNode, ABC):
    """
    A class for the 2D MaxPool node.

    .. tip::
        A MaxPool function is defined as:

        .. math::
            y = \\max \\{ x \\mid x \\in \\text{pool} \\}

    .. attention::
        We only support the 2D MaxPool now.

    .. seealso::
        Refer to
        `torch.nn.MaxPool2d() <https://pytorch.org/docs/stable/generated/torch.nn
        .MaxPool2d.html>`_
        in PyTorch for more arguments' information.

    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation.
    :param act_relax_args: The arguments for the activation relaxation.
    :param kernel_size: The kernel size of the MaxPool.
    :param stride: The stride of the MaxPool.
    :param padding: The padding of the MaxPool.
    :param dilation: The dilation of the MaxPool.
    :param ceil_mode: Whether to use the ceil mode in the MaxPool.s
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        ceil_mode: bool = False,
    ):
        NonLinearNode.__init__(
            self, name, input_names, input_size, shared_data, act_relax_args
        )
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._ceil_mode = ceil_mode

        self._output_size = self._cal_output_size()
        self._no = math.prod(self._output_size)  # number of output elements
        self._ci, self._hi, self._wi = self.input_size  # channel, height, width
        self._co, self._ho, self._wo = self.output_size  # channel, height, width
        self._nk = math.prod(self.output_size[1:])  # number of kernels
        self._nks = math.prod(self.kernel_size)  # number of kernel elements

        self.act_type = ActivationType.MAXPOOL2D

        # Cache
        self._cached_l: Tensor | None = None  # unfolded lower bound
        self._cached_u: Tensor | None = None  # unfolded upper bound
        self._cached_l_max: Tensor | None = None  # max value of each pool
        self._cached_l_argmax: Tensor | None = None  # argmax of each pool
        self._cached_pool_idxs: Tensor | None = None  # pool indexes

        self._degenerated_pool_idxs: list[int] | None = None
        self._trivial_pool_idxs: list[int] | None = None

    def get_nontrivial_neuron_mask(
        self,
        pre_bound: ScalarBound,
        recalculate: bool = False,
        cached: bool = True,
        ignore_degenerate_pool: bool = False,
    ) -> Tensor:
        """
        Get the mask of the neurons that will be handled by nontrivial relaxation.

        .. tip::
            For MaxPool, the trivial neurons are the neurons that always take the same
            input variable as the output variable, i.e., the lower bound of one input
            variable is larger than the upper bound of the other input variables.

        :param pre_bound: The scalar bound of the neurons of the preceding layer.
        :param recalculate: Whether to recalculate the mask.
        :param cached: Whether to cache the mask.
        :param ignore_degenerate_pool: Whether to ignore the degenerated pool, which
            is the pool has input neurons that will never be the maximum value of the
            pool.

        :return: The mask of the neurons that will be handled by non-trivial relaxation.
        """
        if self._cached_nontrivial_mask is not None and not recalculate:
            return self._cached_nontrivial_mask

        pre_l, pre_u = self.get_unfolded_pre_bound(pre_bound, re_calculate=recalculate)
        l_max, l_argmax = self._get_l_argmax(pre_bound, re_calculate=recalculate)
        mask = self._get_nontrivial_neuron_mask(
            pre_l, pre_u, l_max, l_argmax, self.act_relax_args
        )

        if cached:
            self._cached_nontrivial_mask = mask
            self.shared_data.all_nontrivial_neuron_masks[self.name] = mask

        if ignore_degenerate_pool:
            if cached:
                mask = mask.clone()  # Do not change the cached mask.
            mask = torch.where(
                ((pre_u - pre_l) > 0).sum(dim=1) < self._nks, False, mask
            )

        return mask

    @staticmethod
    def _get_nontrivial_neuron_mask(
        pre_l: Tensor,
        pre_u: Tensor,
        l_max: Tensor,
        l_max_arg: Tensor,
        act_relax_args: ActRelaxArgs,
    ) -> Tensor:
        return get_nontrivial_neuron_mask_maxpool2d(
            pre_l, pre_u, l_max, l_max_arg, act_relax_args
        )

    @staticmethod
    def clamp_bounds(bound: ScalarBound) -> ScalarBound:
        return bound

    def get_unfolded_pre_bound(
        self, pre_bound: ScalarBound, re_calculate: bool = False
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cached_l is not None
            and self._cached_u is not None
            and not re_calculate
        ):
            return self._cached_l, self._cached_u

        l, u = unfold_pre_bound_maxpool2d(
            pre_bound.l.reshape(self.input_size),
            pre_bound.u.reshape(self.input_size),
            self._nk,
            self._nks,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

        self._cached_l = l
        self._cached_u = u

        return l, u

    def _get_l_argmax(
        self, pre_bound: ScalarBound, re_calculate: bool = False
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cached_l_max is not None
            and self._cached_l_argmax is not None
            and not re_calculate
        ):
            return self._cached_l_max, self._cached_l_argmax

        pre_l, pre_u = self.get_unfolded_pre_bound(pre_bound)
        l_max, l_argmax = cal_l_argmax_maxpool2d(pre_l, pre_u)
        mask = self._get_nontrivial_neuron_mask(
            pre_l, pre_u, l_max, l_argmax, self.act_relax_args
        )
        # To avoid two neuron have the same max lower bound.
        # But it is the trivial case, the argmax should be the actual max.
        u_argmax = pre_u.argmax(dim=1)
        l_argmax[~mask] = u_argmax[~mask]

        self._cached_l_max = l_max
        self._cached_l_argmax = l_argmax

        return l_max, l_argmax

    def get_pool_idxs(self) -> Tensor:
        """
        Get the input pool indexes.

        :return: The input pool indexes.
        """
        if self._cached_pool_idxs is not None:
            return self._cached_pool_idxs

        # Create an input index tensor
        pool_idxs = get_pool_idxs_maxpool2d(
            self._ci,
            self._hi,
            self._wi,
            self._nks,
            self._nk,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

        self._cached_pool_idxs = pool_idxs

        return pool_idxs

    def clear(self):
        NonLinearNode.clear(self)
        self._cached_l = None
        self._cached_u = None
        self._cached_l_max = None
        self._cached_l_argmax = None
        # self._cached_pool_idxs = {} # This is the same for any input.
        if self._trivial_pool_idxs is not None:
            self._trivial_pool_idxs = None

    def _cal_output_size(self) -> tuple[int, int, int]:

        channel = self.input_size[0]
        height, width, self._output_padding = cal_conv_data_size(
            self.input_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )

        return channel, height, width

    def f(self, x: Tensor) -> Tensor:
        """
        The function of the MaxPool.

        :param x: The input tensor.

        :return: The output tensor.
        """
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )

    @staticmethod
    def df(x: Tensor) -> Tensor:
        """
        The derivative function of the MaxPool.

        :exception RuntimeError: This is only a placeholder.
            MaxPool does not support the derivative function.
        """
        raise RuntimeError(
            "This is only a placeholder. MaxPool does not support the "
            "derivative function."
        )

    @property
    def kernel_size(self) -> tuple:
        """The kernel size of the MaxPool."""
        return self._kernel_size

    @property
    def stride(self) -> tuple:
        """The stride of the MaxPool."""
        return self._stride

    @property
    def padding(self) -> tuple:
        """The padding of the MaxPool."""
        return self._padding

    @property
    def output_padding(self) -> tuple:
        """The output padding of the MaxPool."""
        if self._output_padding is None:
            self._cal_output_size()

        return self._output_padding

    @property
    def dilation(self) -> tuple:
        """The dilation of the MaxPool."""
        return self._dilation

    @property
    def ceil_mode(self) -> bool:
        """The ceil mode of the MaxPool."""
        return self._ceil_mode


class ResidualAddNode(BasicNode, ABC):
    """
    An abstract class for the residual add node.

    .. tip::
        This node is a just a placeholder for the residual add operation and the real
        operation is done in the module structure level.

    :param name: The name of the node.
    :param input_names: The input names of the node.
    :param input_size: The input size of the node.
    :param shared_data: The shared data for bound propagation.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
    ):
        BasicNode.__init__(self, name, input_names, input_size, shared_data)

        self._output_size = self._cal_output_size()
        self._no = math.prod(self.output_size)

    def _cal_output_size(self) -> tuple[int] | tuple[int, int, int]:
        return self.input_size
