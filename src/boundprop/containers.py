"""
This module is used to store the shared data for the bound propagation. The shared data
is used to store the intermediate results of the bound propagation, which can be shared
by different nodes.
"""

__docformat__ = "restructuredtext"
__all__ = ["BPSharedData"]

import logging
from collections import OrderedDict

import torch
from torch import Tensor

from src.utils import *


class BPSharedData:
    """
    This provides a shared data structure for the inequality nodes. When each node
    operates on these data, it can share the data with other nodes.
    For one module, all submodules/nodes share the same data, and it does not affect
    construction of different modules.

    .. tip::
        We separate one hidden layer into two layers, one is the linear layer and the
        other is the non-linear layer.
        The scalar bounds of the non-nonlinear layers can be obtained by the preceding
        linear layers directly, so the scalar bounds of the non-linear layers are not
        commonly calculated for efficiency.

    .. tip::
        The linear relaxation for linear layers are defined by the corresponding weight
        bias, and they are determined by the neural network, so we do not need to store
        the linear relaxation for linear layers.
    """

    def __init__(self, dtype: torch.dtype, device: torch.device):

        self.dtype: torch.dtype = dtype
        """The data type of the shared data."""

        self.device: torch.device = device
        """The device of the shared data."""

        self.all_bounds: dict[str, ScalarBound] = OrderedDict()
        """
        A dictionary that stores the scalar bounds of each layer.
        The key is the name of the layer, and the value is the scalar bounds of the
        layer.
        """

        self.all_relaxations: dict[str, LConstrBound] = OrderedDict()
        """
        A dictionary that stores the linear relaxations of each non-linear layer.
        The key is the name of the layer, and the value is the linear relaxation of the
        layer.
        """

        self.all_nontrivial_neuron_masks: dict[str, Tensor] = OrderedDict()
        """
        A dictionary that stores the masks of the non-trivial neurons.
        The key is the name of the layer, and the value is the mask of the non-trivial
        neurons.
        """

    def clear(self):
        """
        Clear the cache of the layer. Remove the variables that are not needed.

        .. tip::
            This is used to release the calculation resources of one input sample and
            prepare for the next input sample.
        """
        logger = logging.getLogger("rover")
        logger.debug(f"Clear cache of shared variables of bound propagation.")

        self.all_bounds.clear()
        self.all_relaxations.clear()
        self.all_nontrivial_neuron_masks.clear()

    def clear_params(self):
        """
        Clear the optimized parameters of the layer.

        .. tip::
            This is usually used to clear the calculation resources about optimizable
            parameters.
        """
        logger = logging.getLogger("rover")
        logger.debug(f"Clear cache of optimized parameters of bound propagation.")

        for name, bound in self.all_bounds.items():
            self.all_bounds[name] = bound.detach()

        for name, relaxation in self.all_relaxations.items():
            self.all_relaxations[name] = relaxation.detach()
