"""
This module defines the enumerations.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "BoundPropagationMethod",
    "OptimizationMethod",
    "ActivationType",
    "ActRelaxationMode",
    "ConstrTemplate",
    "VerificationStatus",
    "CROWN",
    "DEEPPOLY",
    "ROVER_SN",
]

from enum import Enum


class BoundPropagationMethod(Enum):
    """
    The bound propagation method used in the activation relaxation.
    """

    INEQUALITY = 0
    """ This method construct one lower and one upper ineaquality as the bound of 
    each variable in the neural network. """


class OptimizationMethod(Enum):
    """
    The optimization method used in the activation relaxation.
    """

    LP = 0  # Linear Programming
    """This method verifies a specific property by finding the optimal solution of the 
    linear programming problem."""
    MNLP = 1  # Multi-Neuron LP, kact
    """This method verifies a specific property by finding the optimal solution of the
    linear programming problem with multi-neuron constraints."""


class ActivationType(Enum):
    """
    The activation function type.
    """

    RELU = 0
    """The ReLU activation function."""

    SIGMOID = 1
    """The Sigmoid activation function."""

    TANH = 2
    """The Tanh activation function."""

    ELU = 3
    """The ELU activation function."""

    LEAKY_RELU = 4
    """The Leaky ReLU activation function."""

    MAXPOOL2D = 6
    """The MaxPool2D activation function."""


class ActRelaxationMode(Enum):
    """
    The activation relaxation mode.
    """

    ROVER_SN = 0
    """Single-Neuron Relaxation."""

    CROWN = 7
    """CROWN Relaxation."""

    DEEPPOLY = 8
    """DeepPoly Relaxation."""


CROWN = ActRelaxationMode.CROWN
DEEPPOLY = ActRelaxationMode.DEEPPOLY
ROVER_SN = ActRelaxationMode.ROVER_SN


class ConstrTemplate(Enum):
    """
    The constraint template used in the activation relaxation.
    """

    HEXAGON = 0
    """The hexagon constraint template."""

    RHOMBUS = 2
    """The rhombus constraint template."""

    def __repr__(self):
        return f"{self.name}"


class VerificationStatus(Enum):
    WORKING = 0
    SAT = 1
    UNSAT = 2
    UNKNOWN = 3
    TIMEOUT = 4
    MEMORYOUT = 5

    def __repr__(self):
        return f"{self.name}"
