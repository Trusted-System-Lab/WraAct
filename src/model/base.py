"""
This is the basic module, and it is mainly used to build the bound module from the
neural network file and gives the framework for verification.

We will use the reshape mechanism in Pytorch to internally remove the *flatten* or
*reshape* layer in each node rather than keeping these layers in our bound module.
This is more flexible and reduce considerable complexity in the bound module.
For non-linear node, we always use the flattened shape to handle the operation
considering the complexity of the operation.

We will merge *two neighboring layers into one layer* if possible. This
includes:

- MatMul and Add nodes.
- Gemm and BatchNormalization nodes.
- Sigmoid and Mul nodes for SiLU activation function.

.. attention::
    Currently, we only support the neural network file in the ONNX format.

.. attention::
    Some operations and some arguments may not be supported in the current version.

"""

__docformat__ = "restructuredtext"
__all__ = ["BasicBoundModel"]

import itertools
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Iterator, TypeVar, Generic, Any

import math
import numpy as np
import onnx
import torch
from onnx import numpy_helper, shape_inference
from torch import Tensor
from torch.nn import Module

from src.boundprop import *
from src.utils import *


def _reformat(names: list[str] | str) -> list[str] | str:
    """
    Reformat the names of the submodules. Replace the '.' with '_' because
    torch.nn.Module does not support the submodule name with '.'.

    :param names: The names of the submodules.
    """
    if isinstance(names, str):
        return names.replace(".", "_").replace("/", "_")
    elif isinstance(names, list):
        return [_reformat(name_) for name_ in names]
    else:
        raise TypeError(f"Unsupported type {type(names)} for names.")


T = TypeVar("T", bound=BasicNode)


class BasicBoundModel(Module, ABC, Generic[T]):
    """
    The basic bound module that builds the bound module from the neural network file.

    :param net_file_path: The path of the neural network file.
    :param perturbation_args: The perturbation arguments.
    :param dtype: The data type used in torch.
    :param device: The device used in torch.
    """

    def __init__(
        self,
        net_file_path: str,
        perturbation_args: PerturbationArgs,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super(BasicBoundModel, self).__init__()
        # Check the file is existed.
        if not os.path.exists(net_file_path):
            raise ValueError(f"Model file {net_file_path} does not exist.")

        self._net_file_path = net_file_path
        self._dtype = dtype
        self._device = device
        self._data_settings = {"dtype": dtype, "device": device}
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._input_shape: tuple[int] | tuple[int, int, int] | None = None
        self._output_shape: tuple[int] | None = None
        self._ori_last_weight: Tensor | None = None
        self._ori_last_bias: Tensor | None = None

        self._perturb_args = perturbation_args

        self.submodules: dict[str, T] = OrderedDict()
        """
            All submodules in the bound module. The keys are in a topological order.
        """

    def get_input_bound(self, input_sample: Tensor) -> ScalarBound:
        """
        Get the input lower and upper scalar bound from the input sample.

        :param input_sample: The input sample point.

        :return: The scalar bound (lower and upper bound) of the input sample.
        """
        logger = logging.getLogger("rover")
        logger.debug("Get input bound from the input sample.")

        l, u = self.perturb_args.normalize(input_sample)

        return ScalarBound(l, u).to(**self.data_settings)

    def get_output_weight_bias(
        self, target_label: int, num_labels: int
    ) -> tuple[Tensor, Tensor | None]:
        """
        Get the output weight matrix by the target label and the number of labels.

        .. tip::
            For classification tasks, the difference between the target label and the
            other labels is the objective to verify the robustness of the model, i.e.,
            verify if the lower bound of the target label is higher than the upper
            bound of the other labels.
            This is a linear operation, and we represent it as a matrix multiplication.

        :param target_label: The target label.
        :param num_labels: The number of labels.

        :return: The output weight matrix. For classification tasks, the output bias is
            None.
        """
        logger = logging.getLogger("rover")
        logger.debug(
            f"Get output weight matrix from target label {target_label} of "
            f"total {num_labels} labels."
        )

        output_weight = torch.diag(-torch.ones(num_labels, **self.data_settings))
        output_weight[:, target_label] += 1

        return output_weight, None

    def build(
        self,
        output_weight: Tensor | None = None,
        output_bias: Tensor | None = None,
    ):
        """
        Build the bound module for verification.

        :param output_weight: The output weight matrix.
        :param output_bias: The output bias.
        """
        logger = logging.getLogger("rover")
        logger.debug(f"Start building bound module from {self.net_file_path}.")

        time_total = time.perf_counter()

        logger.debug(f"Load onnx model from {self.net_file_path}.")
        model = onnx.load(self.net_file_path)

        logger.debug("Parse inputs.")
        self._parse_onnx_input(model)
        logger.debug(f"Get input {self.input_name}, shape={self.input_shape}")

        logger.debug("Parse output.")
        self._parse_onnx_output(model)
        logger.debug(f"Get output {self.output_name}, shape={self.output_shape}")

        logger.debug("Parse initializer (parameters and constants).")
        initializers = self._parse_onnx_initializer(model)
        logger.debug(f"Get {len(initializers)} initializers.")

        nodes_iterator = iter(model.graph.node)

        logger.debug(
            "Parse means and stds in onnx model if exists. "
            "If not, use the pre-settings."
        )
        node = self._parse_onnx_mean_std(nodes_iterator)
        logger.debug(
            f"Get mean={self.perturb_args.means.tolist()} and "
            f"std={self.perturb_args.stds.tolist()}"
        )

        logger.debug("Parse node Input.")
        input_names, output_name, input_size = ([], self._input_name, self.input_shape)
        module = self._handle_input(output_name, input_names, input_size)
        self.submodules[output_name] = module
        logger.debug(
            f"Get input={input_names} (shape={input_size}) => "
            f"output={output_name} (shape={module.output_size})."
        )

        constants: dict[str, np.ndarray] = {}
        num_linear_layers = 0
        num_layers = 0
        input_size = module.output_size
        pre_module = module
        module = None
        while node:
            logger.debug(f"Process node {node.op_type}.")
            op_type = node.op_type

            if op_type == "Gemm":
                module = self._parse_gemm(node, input_size, initializers)

            elif op_type == "Conv":
                module = self._parse_conv2d(node, input_size, initializers)

            elif node.op_type in {"Relu", "Sigmoid", "Tanh", "Elu", "Mul", "LeakyRelu"}:
                next_node = None
                if node.op_type == "Sigmoid":
                    next_node = next(nodes_iterator)
                    if next_node.op_type != "Mul":
                        nodes_iterator = itertools.chain((next_node, *nodes_iterator))
                        next_node = None
                module = self._parse_activation(node, input_size, next_node)

            elif node.op_type == "MaxPool":
                module = self._parse_maxpool2d(node, input_size)

            elif node.op_type == "AveragePool":
                module = self._parse_avgpool2d(node, input_size)

            elif node.op_type in {"Flatten", "Reshape"}:
                # Fuse the Flatten and Reshape nodes by specifying the input size.
                self._update_last_submodule_name(node)
                self._parse_flatten_reshape(node, constants)
                input_size = (math.prod(input_size),)

            elif node.op_type == "Add":
                module = self._parse_residual_add(node)

            elif node.op_type == "Upsample":
                module = self._parse_upsample(node, input_size, initializers)

            elif node.op_type == "ConvTranspose":
                module = self._parse_transposeconv2d(node, input_size, initializers)

            elif node.op_type == "Constant":
                constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

            elif op_type == "MatMul":
                node1 = node
                node2 = next(nodes_iterator)
                logger.debug(f"Process node {node2.op_type} together.")

                module = self._parse_matmul_add(node1, node2, input_size, initializers)

            elif node.op_type == "BatchNormalization":
                self._update_last_submodule_name(node)
                self._parse_batchnorm(node, initializers, pre_module)

            elif node.op_type == "Pad":
                self._parse_pad(node, constants)

            else:
                raise NotImplementedError(f"Unsupported op type {node.op_type}")

            if module is not None:
                logger.debug(
                    f"Get "
                    f"input name=[{module.input_names}] (shape={module.input_size}) "
                    f"=> "
                    f"output name={module.name} (shape={module.output_size})."
                )
                logger.debug(f"Build module {module.__class__.__name__} {module.name}")
                num_layers += 1
                if isinstance(module, LinearNode):
                    num_linear_layers += 1

                input_size = module.output_size
                self.submodules[module.name] = module
                pre_module, module = module, None

            node = next(nodes_iterator, None)

        self._update_pre_next_nodes()
        self._check_output_size()
        if self._set_and_add_output_layer(
            output_weight, output_bias, input_size, pre_module
        ):
            num_layers += 1
            num_linear_layers += 1

        logger.debug(
            f"Total {num_linear_layers} linear layers, "
            f"{num_layers - num_linear_layers} non-linear layers."
        )
        logger.debug(
            f"Finish building bound module in {time.perf_counter() - time_total:.4f}s."
        )

    def _update_last_submodule_name(self, onnx_node: onnx.NodeProto):
        """
        Update the last submodule to the new output name of the ONNX node.
        It means skip layers including flatten, reshape, or fuse normalization layers.

        :param onnx_node: The current ONNX node.
        """
        logger = logging.getLogger("rover")

        pre_module = next(reversed(self.submodules.values()))
        new_name = _reformat(onnx_node.output[0])
        logger.debug(
            f"Update the name of the last submodule "
            f"from {pre_module.name} to {new_name}."
        )
        self.submodules.pop(pre_module.name)  # noqa
        pre_module._name = new_name
        self.submodules[pre_module.name] = pre_module  # noqa
        if isinstance(pre_module, InputNode):
            self._input_name = _reformat(pre_module.name)

    def _parse_onnx_input(self, onnx_model: onnx.ModelProto):
        """
        Parse the input information from the ONNX model.

        :param onnx_model: The ONNX model.
        """
        input_node = onnx_model.graph.input[0]
        self._input_name = _reformat(input_node.name)
        self._input_shape = tuple(
            d.dim_value for d in input_node.type.tensor_type.shape.dim
        )

        if len(self._input_shape) not in {2, 4}:
            raise RuntimeError(f"Unsupported input shape: {self._input_shape}.")

        # Remove the dimension of batch.
        self._input_shape = self._input_shape[1:]

    def _parse_onnx_output(self, onnx_model: onnx.ModelProto):
        """
        Parse the output information from the ONNX model.

        :param onnx_model: The ONNX model.
        """
        output_node = onnx_model.graph.output[0]
        self._output_name = _reformat(output_node.name)
        self._output_shape = tuple(
            d.dim_value for d in output_node.type.tensor_type.shape.dim
        )

        if len(self._output_shape) != 2:
            raise RuntimeError(f"Unsupported output shaspe: {self._output_shape}.")

        self._output_shape = self._output_shape[1:]

    def _parse_onnx_initializer(self, onnx_model: onnx.ModelProto) -> dict[str, Tensor]:
        """
        Parse the initializers (parameters) from the ONNX model.

        :param onnx_model: The ONNX model.
        """
        return {
            initializer.name: torch.tensor(
                numpy_helper.to_array(initializer), **self.data_settings
            )
            for initializer in onnx_model.graph.initializer
        }

    def _parse_onnx_mean_std(self, onnx_nodes: Iterator) -> onnx.NodeProto:
        """
        Parse the mean and std from the ONNX model.

        :param onnx_nodes: The iterator of the ONNX nodes.
        """
        logger = logging.getLogger("rover")

        constants = {}
        while True:
            onnx_node = next(onnx_nodes)
            logger.debug(
                f"Try to parse node {onnx_node.op_type} " f"{onnx_node.output[0]}."
            )

            if onnx_node.op_type == "Constant":
                constants[onnx_node.output[0]] = torch.tensor(
                    numpy_helper.to_array(onnx_node.attribute[0].t)
                )
            elif onnx_node.op_type == "Sub":
                self.perturb_args.means = constants[onnx_node.input[1]].squeeze(0)
            elif onnx_node.op_type == "Div":
                self.perturb_args.stds = constants[onnx_node.input[1]].squeeze(0)
            else:
                self._input_name = _reformat(onnx_node.input[0])
                return onnx_node

    @staticmethod
    def _parse_ceil_mode(ceil_mode) -> bool:
        ceil_mode = bool(ceil_mode)

        if ceil_mode:  # Only support ceil_mode=False.
            raise RuntimeError(f"Unsupported ceil_mode {ceil_mode}.")

        return ceil_mode

    @staticmethod
    def _parse_kernel_size(kernel_size) -> tuple:
        kernel_size = tuple(kernel_size)

        if len(kernel_size) != 2 or kernel_size[0] != kernel_size[1]:
            raise RuntimeError(f"Unsupported kernel_size {kernel_size}.")

        return kernel_size

    @staticmethod
    def _parse_dilation(dilation) -> tuple:
        dilation = tuple(dilation)

        if len(dilation) == 4:
            if dilation[0] != dilation[2] or dilation[1] != dilation[3]:
                raise RuntimeError(f"Unsupported dilations {dilation}")
            return tuple(dilation[:2])

        elif len(dilation) == 2:
            if dilation[0] != dilation[1]:
                raise RuntimeError(f"Unsupported dilations {dilation}.")
            return dilation

        else:
            raise RuntimeError(f"Unsupported dilations {dilation}")

    @staticmethod
    def _parse_groups(groups) -> int:
        groups = int(groups)

        if groups != 1:
            raise RuntimeError(f"Unsupported group {groups}")

        return groups

    @staticmethod
    def _parse_padding(padding) -> tuple:
        padding = tuple(padding)

        if len(padding) == 4:
            if padding[0] != padding[2] or padding[1] != padding[3]:
                raise RuntimeError(f"Unsupported paddings {padding}.")
            return tuple(padding[:2])

        elif len(padding) == 2:
            if padding[0] != padding[1]:
                raise RuntimeError(f"Unsupported paddings {padding}.")
            return padding

        else:
            raise RuntimeError(f"Unsupported paddings {padding}")

    @staticmethod
    def _parse_stride(stride) -> tuple:
        stride = tuple(stride)
        if len(stride) == 4:
            if stride[0] != stride[2] or stride[1] != stride[3]:
                raise RuntimeError(f"Unsupported strides {stride}.")
            stride = tuple(stride[:2])

        elif len(stride) == 2:
            if stride[0] != stride[1]:
                raise RuntimeError(f"Unsupported strides {stride}.")

        else:
            raise RuntimeError(f"Unsupported strides {stride}")

        if any(s == 0 for s in stride):
            raise RuntimeError(f"Unsupported strides {stride}")

        return stride

    @staticmethod
    def _parse_count_include_pad(count_include_pad) -> bool:
        count_include_pad = bool(count_include_pad)

        if not count_include_pad:
            raise RuntimeError(f"Unsupported count_include_pad {count_include_pad}")

        return count_include_pad

    @staticmethod
    def _parse_conv_kwargs(
        onnx_node: onnx.NodeProto,
    ) -> dict[str, tuple | int]:
        """
        Parse the convolutional arguments from the ONNX node.

        :param onnx_node: The ONNX node.

        :return: The convolutional arguments.
        """
        logger = logging.getLogger("rover")

        groups = 1
        dilation = (1, 1)
        kernel_size = (1, 1)  # Just check # noqa
        padding = (0, 0)
        stride = (1, 1)
        if onnx_node.op_type in {"Conv", "ConvTranspose"}:
            logger.debug("Parse conv arguments.")
            for attr in onnx_node.attribute:
                if attr.name == "dilations":
                    dilation = BasicBoundModel._parse_dilation(attr.ints)
                elif attr.name == "kernel_shape":
                    # Just check
                    kernel_size = BasicBoundModel._parse_kernel_size(attr.ints)  # noqa
                elif attr.name == "pads":
                    padding = BasicBoundModel._parse_padding(attr.ints)
                elif attr.name == "strides":
                    stride = BasicBoundModel._parse_stride(attr.ints)
                elif attr.name == "group":
                    groups = BasicBoundModel._parse_groups(attr.i)
                else:
                    raise NotImplementedError(f"Unsupported attribute {attr.name}")
        kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

        logger.debug(f"Get conv kwargs {kwargs}")

        return kwargs

    @staticmethod
    def _parse_pool_kwargs(
        onnx_node: onnx.NodeProto,
    ) -> dict[str, tuple | int]:
        """
        Parse the pooling arguments from the ONNX node.

        :param onnx_node: The ONNX node.

        :return: The pooling arguments
        """
        logger = logging.getLogger("rover")

        ceil_mode = False
        dilation = (1, 1)
        kernel_size = (1, 1)
        padding = (0, 0)
        stride = (1, 1)
        count_include_pad = True  # For AveragePool # noqa
        if onnx_node.op_type in {"MaxPool", "AveragePool"}:
            logger.debug("Parse pool arguments.")
            for attr in onnx_node.attribute:
                if attr.name == "ceil_mode":
                    ceil_mode = BasicBoundModel._parse_ceil_mode(attr.i)
                elif attr.name == "dilations":
                    dilation = BasicBoundModel._parse_dilation(attr.ints)
                elif attr.name == "kernel_shape":
                    kernel_size = BasicBoundModel._parse_kernel_size(attr.ints)
                elif attr.name == "pads":
                    padding = BasicBoundModel._parse_padding(attr.ints)
                elif attr.name == "strides":
                    stride = BasicBoundModel._parse_stride(attr.ints)
                elif attr.name == "count_include_pad":
                    # Just check
                    pass
                    # count_include_pad = BasicBoundModel._parse_count_include_pad(
                    #     attr.i
                    # )  # noqa
                else:
                    raise NotImplementedError(f"Unsupported attribute {attr.name}")

        kwargs = {
            "dilation": dilation,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
        }

        logger.debug(f"Get pool kwargs {kwargs}")

        return kwargs

    def _update_pre_next_nodes(self):
        logger = logging.getLogger("rover")
        logger.debug("Set the pre and next nodes for each module.")

        # Set the pre and next nodes for each module.
        modules_iter = iter(self.submodules.values())

        # For the first input layer, it has no pre nodes, and we set it None.
        module = next(modules_iter)
        module.pre_nodes = None
        module.next_nodes = []

        module = next(modules_iter, None)
        while module is not None:
            module.pre_nodes = []
            module.next_nodes = []
            module = next(modules_iter, None)

        # For the last output layer, it has no next nodes, and we set it None.
        module = next(reversed(self.submodules.values()))
        module.next_nodes = None

        for name, module in self.submodules.items():
            for input_name in module.input_names:  # type: ignore
                pre_module = self.submodules[input_name]
                module.pre_nodes.append(pre_module)
                pre_module.next_nodes.append(module)  # type: ignore

        logger.debug("Finish setting the pre and next nodes for each module.")

    def _check_output_size(self):
        """
        Check the output size of each module in the bound module by comparing with the
        inferred shape from the ONNX model.
        """
        logger = logging.getLogger("rover")
        logger.debug("Check output size of each module.")

        # Check the shape is correct.
        model = onnx.load(self.net_file_path)
        inferred_model = shape_inference.infer_shapes(model)

        for value_info in inferred_model.graph.value_info:
            output_size = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
            output_size = tuple(output_size[1:])  # Remove the batch dimension.

            name = _reformat(value_info.name)
            submodule = self.submodules.get(name, None)
            if submodule is None:
                continue
            our_output_size = submodule.output_size

            # This is for flatten layer.
            equal_neurons = math.prod(our_output_size) == math.prod(output_size)  # noqa
            if our_output_size != output_size and not equal_neurons:
                raise RuntimeError(
                    f"Output size {our_output_size} of "
                    f"module {name} is not equal to {output_size}."
                )

        logger.debug("Output size of each module is correct.")

    def update_output_constrs(self, weight: Tensor, bias: Tensor | None = None):
        """
        Update the last linear layer with the output weight matrix and bias.

        .. tip::
            We will try to merge two neighboring linear layers into one layer.

        1. If the last layer of the neural network is a linear layer, we will update the
           weight and bias of the last layer by merging the output weight matrix and
           bias.
        2. Otherwise, we will add a new linear layer to the neural network taking the
           output weight and bias as the linear layer's weight and bias..

        """

        logger = logging.getLogger("rover")
        logger.debug("Update output constraints for the last layer.")

        last_module = self.submodules[self.output_name]
        if not isinstance(last_module, GemmNode):
            raise RuntimeError(f"Unsupported last layer type {type(last_module)}.")

        if self._ori_last_weight is not None and self._ori_last_bias is not None:
            last_module.weight = weight @ self._ori_last_weight
            if self._ori_last_bias is not None and bias is not None:
                last_module.bias = weight @ self._ori_last_bias + bias
            elif self._ori_last_bias is not None:
                last_module.bias = weight @ self._ori_last_bias
            else:  # bias is not None:
                last_module.bias = bias

        else:
            last_module.weight = weight
            if bias is not None:
                last_module.bias = bias

    def restore_output_constraints(self):
        """
        Restore the original weight and bias of the last layer, i.e., remove the
        output weight matrix and bias from the merged layer.
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
            logger.debug("No need to restore the output constraints.")

    @staticmethod
    def _parse_names(node: onnx.NodeProto):
        # Only parse the first input and output names.
        input_names = _reformat([node.input[0]] if len(node.input) > 0 else [])
        name = _reformat(node.output[0])
        return input_names, name

    def _parse_gemm(
        self,
        node: onnx.NodeProto,
        input_size: tuple[int],
        initializers: dict[str, Tensor],
    ) -> GemmNode:
        """
        Parse the Gemm node.

        :param node: The Gemm node.
        :param initializers: The initializers.

        :return: The weight and bias of the Gemm node.
        """
        input_names, name = self._parse_names(node)

        weight = torch.asarray(initializers[node.input[1]], **self.data_settings)
        bias = torch.asarray(initializers[node.input[2]], **self.data_settings)
        args = (name, input_names, input_size, weight, bias)

        return self._handle_gemm(*args)

    def _parse_conv2d(
        self,
        node: onnx.NodeProto,
        input_size: tuple[int, int, int],
        initializers: dict[str, Tensor],
    ) -> Conv2DNode:
        """
        Parse the Conv node.

        :param node: The Conv node.
        :param initializers: The initializers.

        :return: The weight and bias of the Conv node.
        """
        input_names, name = self._parse_names(node)

        weight = torch.asarray(initializers[node.input[1]], **self.data_settings)
        bias = torch.asarray(initializers[node.input[2]], **self.data_settings)
        args = (name, input_names, input_size, weight, bias)
        conv_kwargs = self._parse_conv_kwargs(node)

        return self._handle_conv2d(*args, **conv_kwargs)

    def _parse_matmul_add(
        self,
        node1: onnx.NodeProto,
        node2: onnx.NodeProto,
        input_size: tuple[int],
        initializers: dict[str, Tensor],
    ) -> GemmNode:
        """
        Parse the MatMul and Add node.

        :param node1: The MatMul node.
        :param node2: The Add node.
        :param initializers: The initializers.

        :return: The weight and bias of the Gemm node.
        """
        input_names = _reformat([node1.input[1]])
        output_name = _reformat(node2.output[0])

        weight = torch.asarray(initializers[node1.input[0]], **self.data_settings)
        bias = torch.asarray(initializers[node2.input[1]], **self.data_settings)
        args = (input_names, output_name, input_size, weight, bias)

        return self._handle_gemm(*args)

    def _parse_activation(
        self,
        node: onnx.NodeProto,
        input_size: tuple[int] | tuple[int, int, int],
        next_node: onnx.NodeProto = None,
    ) -> BasicNode:
        """
        Parse the activation node.

        :param node: The activation node.

        :return: The activation node.
        """
        logger = logging.getLogger("rover")
        input_names, name = self._parse_names(node)
        op_type = node.op_type

        if op_type == "Relu":
            return self._handle_relu(name, input_names, input_size)

        elif op_type == "Sigmoid":
            if next_node is not None:
                logger.debug(f"Get a SiLU activation funciton.")
                name = _reformat(next_node.output[0])
                return self._handle_silu(name, input_names, input_size)
            return self._handle_sigmoid(name, input_names, input_size)

        elif op_type == "Tanh":
            return self._handle_tanh(name, input_names, input_size)

        elif op_type == "Elu":
            alpha = float(node.attribute[0].f)

            if alpha != 1.0:
                raise RuntimeError(f"Unsupported alpha {alpha} for Elu function.")

            return self._handle_elu(name, input_names, input_size)

        elif op_type == "LeakyRelu":
            alpha = float(node.attribute[0].f)

            if abs(alpha - 0.01) > 1e-6:
                raise RuntimeError(f"Unsupported alpha {alpha} for LeakyReLU function.")

            return self._handle_leakyrelu(name, input_names, input_size)

        else:
            raise NotImplementedError(f"Unsupported activation {op_type}.")

    @staticmethod
    def _parse_flatten_reshape(node: onnx.NodeProto, constants: dict[str, np.ndarray]):
        """
        Parse the Flatten and Reshape node.

        :param node: The Flatten or Reshape node.
        """
        if node.op_type == "Reshape":
            allowzero = bool(node.attribute[0].i)
            if allowzero:
                raise RuntimeError(f"Unsupported reshape with allowzero {allowzero}.")

            output_size = tuple(constants[node.input[1]])[1:]
            if output_size != (-1,):
                raise RuntimeError(f"Unsupported reshape with output {output_size}.")

    def _parse_maxpool2d(
        self, node: onnx.NodeProto, input_size: tuple[int, int, int]
    ) -> MaxPool2DNode:
        """
        Parse the MaxPool node.

        :param node: The MaxPool node.
        :param initializers: The initializers.

        :return: The MaxPool node.
        """
        input_names, name = self._parse_names(node)
        args = (name, input_names, input_size)

        pool_kwargs = self._parse_pool_kwargs(node)

        return self._handle_maxpool2d(*args, **pool_kwargs)

    def _parse_residual_add(self, node: onnx.NodeProto) -> ResidualAddNode:
        """
        Parse the residual Add node.

        :param node: The residual Add node.
        """
        logger = logging.getLogger("rover")
        input_names = _reformat(list(node.input))  # We need all input names.

        if len(input_names) != 2:
            raise RuntimeError(
                f"Unsupported residual block with {len(input_names)} inputs."
            )

        for name_ in input_names:
            if name_ not in self.submodules.keys():
                raise RuntimeError(
                    f"Unsupported residual block with input {name_} not found."
                )

        name = _reformat(node.output[0])

        def find_residual_block_input(names: list[str]) -> str:
            names_set = set(names)
            names = reversed(self.submodules.keys())
            while True:
                pre_name = next(names)
                input_name = self.submodules[pre_name].input_names[0]  # type: ignore
                if input_name in names_set:
                    return input_name
                names_set.add(input_name)

        residual_input_name = find_residual_block_input(input_names)
        logger.debug(f"Find input {residual_input_name} of residual block.")

        def update_residual_path_input_size():
            # We only need to consider the second path of the residual block.
            # It has wrong input sizes because, by default, the next submodule will use
            # the output size of the previous submodule.

            # Get all modules in the residual block.
            modules_to_update = []
            modules = reversed(self.submodules.values())
            module = next(modules)
            while module.name != residual_input_name:
                modules_to_update.append(module)
                module = next(modules)
            modules_to_update.reverse()

            for module in modules_to_update:
                if len(module.input_names) != 1:  # type: ignore
                    raise RuntimeError(
                        f"Unsupported residual block with multiple inputs {module}."
                    )
                input_size_ = self.submodules[module.input_names[0]].output_size  # noqa
                logger.debug(
                    f"Update input size of {module} "
                    f"from {module.input_size} to {input_size_}."
                )
                module.input_size = input_size_

            return module.output_size

        logger.debug("Update residual path input size.")
        input_size = update_residual_path_input_size()

        return self._handle_residual_add(name, input_names, input_size)  # noqa

    def _set_and_add_output_layer(
        self,
        weight: Tensor | None,
        bias: Tensor | None,
        input_size: tuple[int],
        pre_module: BasicNode,
    ):
        """
        Set the output layer for the bound module.
        :param weight: The output weight matrix.
        :param bias: The output bias.
        :param input_size: The input size of the output layer.
        :param pre_module: The previous module.
        :return: True if the output layer is added, False if the output constraints
            are merged in the last layer.
        """
        logger = logging.getLogger("rover")
        if weight is None:
            weight = torch.eye(self.output_shape[0], **self.data_settings)

        if isinstance(pre_module, GemmNode):
            logger.debug(
                f"Fuse output constraints with the last linear layer {pre_module}."
            )
            self._ori_last_weight = pre_module.weight.clone()
            self._ori_last_bias = pre_module.bias.clone()

            fused_weight = weight @ pre_module.weight
            fused_bias = weight @ pre_module.bias
            if bias is not None:
                fused_bias += bias
            pre_module.weight = fused_weight
            pre_module.bias = fused_bias

            return False
        else:
            logger.debug(f"Add a new Gemm layer for the output constraints.")
            name = _reformat("verified_output")
            input_names = [_reformat(self.output_name)]
            self._output_name = name
            args = (name, input_names, input_size, weight, bias)
            module = self._handle_gemm(*args)
            self.submodules[name] = module
            pre_module.next_nodes = [module]
            module.pre_nodes = [pre_module]

            return True

    def forward(
        self,
        target_label: int,
        input_sample: Tensor,
        input_bound: ScalarBound,
        *args,
        **kwargs,
    ) -> ScalarBound:
        """
        Forward the input sample through the neural network. This is a placeholder.

        :param target_label: The target label.
        :param input_sample: The input sample.
        :param input_bound: The scalar bound of the input sample.

        :return: The scalar bound of the output.
        """
        raise NotImplementedError(
            "The forward method should be implemented in the subclass."
        )

    @abstractmethod
    def _handle_input(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> InputNode:
        pass

    @abstractmethod
    def _handle_gemm(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int],
        weight: Tensor,
        bias: Tensor | None = None,
    ) -> GemmNode:
        pass

    @abstractmethod
    def _handle_conv2d(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        weight: Tensor,
        bias: Tensor = None,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        ceil_mode: bool = False,
    ) -> Conv2DNode:
        pass

    @abstractmethod
    def _handle_maxpool2d(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        ceil_mode: bool = False,
    ) -> MaxPool2DNode:
        pass

    @abstractmethod
    def _handle_relu(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> ReLUNode:
        pass

    @abstractmethod
    def _handle_sigmoid(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> SigmoidNode:
        pass

    @abstractmethod
    def _handle_tanh(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> TanhNode:
        pass

    @abstractmethod
    def _handle_elu(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> ELUNode:
        pass

    @abstractmethod
    def _handle_leakyrelu(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> LeakyReLUNode:
        pass

    @abstractmethod
    def _handle_residual_add(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> ResidualAddNode:
        pass

    @property
    def perturb_args(self):
        """The perturbation arguments."""
        return self._perturb_args

    @property
    def net_file_path(self) -> str:
        """The path of the neural network file."""
        return self._net_file_path

    @property
    def input_name(self) -> str:
        """The name of the input layer."""
        return self._input_name

    @property
    def output_name(self) -> str:
        """The name of the output layer."""
        return self._output_name

    @output_name.setter
    def output_name(self, name):
        """Set the name of the output layer."""
        self._output_name = name

    @property
    def input_shape(self) -> tuple[int] | tuple[int, int, int]:
        """The shape of the input layer."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int]:
        """The shape of the output layer."""
        return self._output_shape

    @property
    def dtype(self) -> torch.dtype:
        """The data type used in torch."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """The device used in torch."""
        return self._device

    @property
    def data_settings(self) -> dict[str, Any]:
        """The data settings used in torch, i.e., dtype and device."""
        return self._data_settings
