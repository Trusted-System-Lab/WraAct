from torch import Tensor

from src.utils import ActRelaxArgs

__docformat__ = ["restructuredtext"]
__all__ = [
    "get_nontrivial_neuron_mask_relu",
    "get_nontrivial_neuron_mask_sigmoid",
    "get_nontrivial_neuron_mask_tanh",
    "get_nontrivial_neuron_mask_elu",
    "get_nontrivial_neuron_mask_leakyrelu",
    "get_nontrivial_neuron_mask_silu",
    "get_nontrivial_neuron_mask_maxpool2d",
]


def get_nontrivial_neuron_mask_relu(
    pre_l: Tensor,
    pre_u: Tensor,
    l: Tensor,
    u: Tensor,
    act_relax_args: ActRelaxArgs,
) -> Tensor:
    r_min_half = act_relax_args.min_half_range
    r_min = act_relax_args.min_range
    r = -pre_l * pre_u

    return (pre_l < -r_min_half) & (pre_u > r_min_half) & (r > r_min)


def get_nontrivial_neuron_mask_sigmoid(
    pre_l: Tensor,
    pre_u: Tensor,
    l: Tensor,
    u: Tensor,
    act_relax_args: ActRelaxArgs,
) -> Tensor:
    r_min = act_relax_args.min_range
    limit = act_relax_args.sigmoid_limit_bound
    r = (u - l) * (pre_u - pre_l)

    return (pre_l < limit) & (pre_u > -limit) & (r > r_min)


def get_nontrivial_neuron_mask_tanh(
    pre_l: Tensor,
    pre_u: Tensor,
    l: Tensor,
    u: Tensor,
    act_relax_args: ActRelaxArgs,
) -> Tensor:
    r_min = act_relax_args.min_range
    limit = act_relax_args.tanh_limit_bound
    r = (u - l) * (pre_u - pre_l)

    return (pre_l < limit) & (pre_u > -limit) & (r > r_min)


def get_nontrivial_neuron_mask_elu(
    pre_l: Tensor,
    pre_u: Tensor,
    l: Tensor,
    u: Tensor,
    act_relax_args: ActRelaxArgs,
) -> Tensor:
    r_min_half = act_relax_args.min_half_range
    r_min = act_relax_args.min_range
    r = (u - l) * (pre_u - pre_l)

    return (pre_l < -r_min_half) & (pre_u > r_min_half) & (r > r_min)


def get_nontrivial_neuron_mask_leakyrelu(
    pre_l: Tensor,
    pre_u: Tensor,
    l: Tensor,
    u: Tensor,
    act_relax_args: ActRelaxArgs,
) -> Tensor:
    r_min_half = act_relax_args.min_half_range
    r_min = act_relax_args.min_range
    r = -pre_l * pre_u

    return (pre_l < -r_min_half) & (pre_u > r_min_half) & (r > r_min)


def get_nontrivial_neuron_mask_silu(
    pre_l: Tensor,
    pre_u: Tensor,
    l: Tensor,
    u: Tensor,
    act_relax_args: ActRelaxArgs,
) -> Tensor:
    r_min_half = act_relax_args.min_half_range
    r_min = act_relax_args.min_range

    return (
        (pre_l < -r_min_half)
        & (pre_u > r_min_half)
        & ((u - l) * (pre_u - pre_l) > r_min)
    )


def get_nontrivial_neuron_mask_maxpool2d(
    pre_l: Tensor,
    pre_u: Tensor,
    pre_l_max: Tensor,
    pre_l_argmax: Tensor,
    act_relax_args: ActRelaxArgs,
) -> Tensor:
    # Get the second max upper bound in each kernel.
    return pre_l_max < pre_u.topk(2, dim=1).values[:, 1]
