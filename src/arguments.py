"""
This module contains the arguments for the experiment.
"""

__docformat__ = "restructuredtext"
__all__ = ["Arguments"]

import logging
import os
from dataclasses import dataclass

import torch

from src.utils import *


@dataclass
class Arguments:
    """
    This is the data class for the arguments of the experiment.
    """

    net_file_path: str
    """The path of the network file."""

    dataset: str
    """The dataset name."""

    perturbation_radius: float
    """The perturbation radius."""

    bound_propagation_method: BoundPropagationMethod = BoundPropagationMethod.INEQUALITY
    """
    The bound propagation method.
    
    Refer to :class:`BoundPropagationMethod` for more details.
    """

    act_relax_mode: ActRelaxationMode = ActRelaxationMode.ROVER_SN
    """
    The activation relaxation mode.
    
    Refer to :class:`ActRelaxationMode` for more details.
    """

    optimization_method: OptimizationMethod | None = None
    """
    The optimization method.
    
    Refer to :class:`OptimizationMethod` for more details.
    """

    num_labels: int | None = None
    """The number of labels."""

    check_ignored_samples: bool = True
    """
    Whether to check ignored samples, which are samples that are not classified 
    correctly by the given model.
    """

    num_samples: int = 1
    """The number of samples to verify."""

    first_sample_index: int = 0
    """The index of the first sample to verify."""

    normalize: bool = True
    """Whether to normalize the input data."""

    means: torch.Tensor = None
    """The means of normalization."""

    stds: torch.Tensor = None
    """The standard deviations of normalization."""

    input_limited_range: tuple = (0.0, 1.0)
    """
    The limited range of the input data.
     
    For example, an image with a pixel value range of [0, 1] should be limited to [0, 1]
    after perturbed because the perturbation should not exceed the pixel value range.
    """

    net_dir_path: str | None = None
    """The directory path of the network file."""

    net_file_name: str | None = None
    """The name of the network file, excluding the file extension and directory path."""

    log_file: str | None = None
    """The name of the log file."""

    log_level: int = logging.INFO
    """The logging level."""

    random_seed: int = 0
    """The random seed for all random number generators (python, numpy, torch)."""

    dtype: str = "float64"
    """The data type that numpy and torch use."""

    device: str = "cpu"
    """The device that torch uses."""

    perturbation_args: PerturbationArgs | None = None
    """The perturbation arguments."""

    act_relax_args: ActRelaxArgs | None = None
    """The activation relaxation arguments."""

    lp_args: LPArgs | None = None
    """The LP arguments."""

    kact_lp_args: KActLPArgs | None = None
    """The kact LP arguments."""

    use_adv_attack: bool = False

    def __post_init__(self):

        if not os.path.exists(self.net_file_path):
            raise ValueError(f"Network file {self.net_file_path} does not exist.")

        if not self.net_file_path.endswith(".onnx"):
            raise ValueError(f"Network file {self.net_file_path} is not an ONNX file.")

        if self.dataset not in {"mnist", "cifar10"}:
            raise ValueError(f"Dataset {self.dataset} is not supported.")

        if self.perturbation_radius < 0:
            raise ValueError(
                f"Perturbation radius {self.perturbation_radius} should be "
                f"non-negative."
            )

        if self.num_labels is not None:
            if self.num_labels < 2:
                raise ValueError(
                    f"Number of labels {self.num_labels} should be at least 2."
                )

        if self.num_samples < 1:
            raise ValueError(
                f"Number of samples {self.num_samples} should be at least 1."
            )

        if self.first_sample_index < 0:
            raise ValueError(
                f"First sample index {self.first_sample_index} should be non-negative."
            )

        if self.input_limited_range[0] > self.input_limited_range[1]:
            raise ValueError(
                f"Input limited range {self.input_limited_range} is invalid."
            )

        if self.dataset in {"mnist", "cifar10"}:
            self.num_labels = 10

        if self.net_dir_path is None:
            self.net_dir_path = os.path.dirname(self.net_file_path)

        if self.net_file_name is None:
            self.net_file_name = os.path.basename(self.net_file_path).split(".")[0]

        # If the log file is not None, check the dir path.
        if self.log_file is not None:
            log_dir_path = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir_path):
                os.makedirs(log_dir_path)

        _build_logger(self.log_file, self.log_level)
        self._set_means_stds()
        self._set_args()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _set_args(self):

        logger = logging.getLogger("rover")
        # -------- Set perturbation arguments --------

        dtype = torch.float64 if self.dtype == "float64" else torch.float32
        device = torch.device(self.device)

        self.perturbation_args = PerturbationArgs(
            epsilon=self.perturbation_radius,
            norm=float("inf"),
            means=self.means.to(dtype=dtype, device=device),
            stds=self.stds.to(dtype=dtype, device=device),
            lower_limit=self.input_limited_range[0],
            upper_limit=self.input_limited_range[1],
        )

        logger.debug(f"Set perturbation arguments: {self.perturbation_args}.")
        # -------- Set activation relaxation arguments --------
        self.act_relax_args = ActRelaxArgs(mode=self.act_relax_mode)
        self.act_relax_args.update_scalar_bounds_per_layer = (
            self.act_relax_mode != CROWN
        )
        logger.debug(f"Set activation relaxation arguments: {self.act_relax_args}.")

        # -------- Set LP arguments --------
        if self.optimization_method is not None:
            self.lp_args = LPArgs()
            logger.debug(f"Set LP arguments: {self.lp_args}")

            if self.optimization_method == OptimizationMethod.MNLP:
                self.kact_lp_args = KActLPArgs()
                logger.debug(f"Set kact LP arguments: {self.kact_lp_args}")

    def __str__(self) -> str:
        return (
            f"Arguments(\n"
            + ",\n".join(f"\t{k:<25}: {v}" for k, v in self.__dict__.items())
            + "\n"
            f")"
        )

    def _set_means_stds(self):
        """
        Set the means and standard deviations for normalization.
        """
        if self.means is None or self.stds is None:
            if self.dataset == "mnist":
                self.means = torch.tensor([0.0])
                self.stds = torch.tensor([1.0])
            elif self.dataset == "cifar10":
                self.means = torch.tensor([[[0.4914]], [[0.4822]], [[0.4465]]])
                self.stds = torch.tensor([[[0.2023]], [[0.1994]], [[0.201]]])
            else:
                raise ValueError(f"Dataset {self.dataset} is not supported.")

        if self.net_file_name in {
            "mnist_sigmoid_6_500",
            "mnist_tanh_6_500",
            "mnist_sigmoid_convmed",
            "mnist_tanh_convmed",
            "mnist_convSmallRELU__Point",
        }:
            self.means = torch.tensor([0.1307])
            self.stds = torch.tensor([0.30810001])

        self.normalize = True
        if self.net_file_name in {
            "mnist_sigmoid_6_500",
            "mnist_tanh_6_500",
            "mnist_sigmoid_convmed",
            "mnist_tanh_convmed",
            "mnist_convSmallRELU__Point",
            "cifar_sigmoid_6_500",
            "cifar_tanh_6_500",
            "cifar_sigmoid_convmed",
            "cifar_tanh_convmed",
        }:
            self.normalize = False


def _build_logger(log_name: str | None, log_level=logging.INFO):

    logger = build_logger(
        LoggerArgs(log_level=log_level, log_file=log_name, log_console=True),
        name="rover",
    )

    logger.debug("Setup logger.")
