"""This is for calculating the function hull with given constraints and bounds."""

import os
import sys
import time
import warnings

cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, "../../../WraAct")
sys.path.insert(0, "../../../ELINA/python_interface/")

import numpy as np

try:
    from fconv import ftanh_orthant, fsigm_orthant, fkpool  # noqa
except ImportError as e:
    warnings.warn(
        f"[WARNING] ELINA is not installed, so we cannot use some methods in fconv: {e}"
    )


from src.funchull.acthull import *
from src.funchull.ablation_study import *


def read_constraints_and_bounds(constraints_file_path: str, bounds_file_path: str):
    with open(constraints_file_path, "r") as f:
        constraints = f.readlines()
    with open(bounds_file_path, "r") as f:
        bounds = f.readlines()
    constraints = [eval(constraint) for constraint in constraints]
    bounds = [eval(bound) for bound in bounds]
    return constraints, bounds


if __name__ == "__main__":
    time_total = time.perf_counter()

    print("[INFO] Start calculating the function hulls...")

    # Get the current directory and go to the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    constraints_dir = os.path.join(current_dir, "../polytope_samples")
    bounds_dir = os.path.join(current_dir, "../polytope_bounds")
    constraints_files = os.listdir(constraints_dir)
    constraints_files = [file for file in constraints_files if file.endswith(".txt")]
    # Sort by dimension
    constraints_files.sort(key=lambda x: int(x.split(".")[-2].split("_")[1][:-1]))

    for method in [
        "single_sigmoid",
        "single_tanh",
        "single_maxpool",
        "single_leakyrelu",
        "single_elu",
        "prima_sigmoid",
        "prima_tanh",
        "prima_maxpool",
        "our_sigmoid",
        "our_tanh",
        "our_maxpool_dlp",
        "our_leakyrelu",
        "our_elu",
        "our_sigmoid-a",
        "our_sigmoid-b",
        "our_tanh-a",
        "our_tanh-b",
        "our_elu-a",
        "our_maxpool_dlp-a",
    ]:
        for i, constraints_file in enumerate(constraints_files):
            if "prima" in method and constraints_file.endswith("oct.txt"):
                dim = int(constraints_file.split(".")[-2].split("_")[-3][:-1])
                constraints_file_path = os.path.join(constraints_dir, constraints_file)
                bounds_file_path = os.path.join(
                    bounds_dir, constraints_file.replace("_oct.txt", "_bounds.txt")
                )
            elif "prima" not in method and not constraints_file.endswith("oct.txt"):
                dim = int(constraints_file.split(".")[-2].split("_")[-2][:-1])
                constraints_file_path = os.path.join(constraints_dir, constraints_file)
                bounds_file_path = os.path.join(
                    bounds_dir, constraints_file.replace(".txt", "_bounds.txt")
                )

            else:
                print(f"[INFO] Skipping {constraints_file} with {method}...")

                continue

            if dim > 4 and (
                "prima" in method
                or "-a" in method
                or "-b" in method
                or "single" in method
            ):
                # PRIMA does not support dimensions greater than 4
                continue

            print(f"[INFO] Processing {constraints_file_path} with {method}...")

            constraints_list, bounds_list = read_constraints_and_bounds(
                constraints_file_path, bounds_file_path
            )

            bounds_file_path = os.path.basename(bounds_file_path)
            saved_file_path = bounds_file_path.replace(
                "_bounds.txt", f"_{method}.txt"
            ).split("\\")[-1]

            file = open(saved_file_path, "w")
            for n, (constraints, bounds) in enumerate(
                zip(constraints_list, bounds_list)
            ):
                lb, ub = bounds
                lb = np.asarray(lb)
                ub = np.asarray(ub)

                constraints = np.asarray(constraints)
                d = constraints.shape[1] - 1

                time_cal = time.perf_counter()
                output_constraints = None
                if method == "prima_sigmoid":
                    try:
                        output_constraints = fsigm_orthant(constraints)
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to calculate hull for "
                            f"{constraints_file_path} with {method}: {e}"
                        )

                elif method == "prima_tanh":
                    try:
                        output_constraints = ftanh_orthant(constraints)
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to calculate hull for "
                            f"{constraints_file_path} with {method}: {e}"
                        )
                elif method == "prima_maxpool":
                    try:
                        output_constraints = fkpool(constraints)  # noqa
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to calculate hull for "
                            f"{constraints_file_path} with {method}: {e}"
                        )
                elif method == "our_sigmoid":
                    fun_hull = SigmoidHull(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_tanh":
                    fun_hull = TanhHull(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_maxpool_dlp":
                    fun_hull = MaxPoolHullDLP(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_leakyrelu":
                    fun_hull = LeakyReLUHull(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_elu":
                    fun_hull = ELUHull(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_sigmoid-a":
                    fun_hull = SigmoidHullA(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_sigmoid-b":
                    fun_hull = SigmoidHullB(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_tanh-a":
                    fun_hull = TanhHullA(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_tanh-b":
                    fun_hull = TanhHullB(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_elu-a":
                    fun_hull = ELUHullA(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "our_maxpool_dlp-a":
                    fun_hull = MaxPoolDLPHullA(if_cal_single_neuron_constrs=True)
                    output_constraints = fun_hull.cal_hull(input_constrs=constraints)
                elif method == "single_sigmoid":
                    fun_hull = SigmoidHull(
                        if_cal_single_neuron_constrs=True,
                        if_cal_multi_neuron_constrs=False,
                    )
                    output_constraints = fun_hull.cal_hull(
                        input_lower_bounds=lb, input_upper_bounds=ub
                    )
                elif method == "single_tanh":
                    fun_hull = TanhHull(
                        if_cal_single_neuron_constrs=True,
                        if_cal_multi_neuron_constrs=False,
                    )
                    output_constraints = fun_hull.cal_hull(
                        input_lower_bounds=lb, input_upper_bounds=ub
                    )
                elif method == "single_maxpool":
                    fun_hull = MaxPoolHullDLP(
                        if_cal_single_neuron_constrs=True,
                        if_cal_multi_neuron_constrs=False,
                    )
                    output_constraints = fun_hull.cal_hull(
                        input_lower_bounds=lb, input_upper_bounds=ub
                    )
                elif method == "single_leakyrelu":
                    fun_hull = LeakyReLUHull(
                        if_cal_single_neuron_constrs=True,
                        if_cal_multi_neuron_constrs=False,
                    )
                    output_constraints = fun_hull.cal_hull(
                        input_lower_bounds=lb, input_upper_bounds=ub
                    )
                elif method == "single_elu":
                    fun_hull = ELUHull(
                        if_cal_single_neuron_constrs=True,
                        if_cal_multi_neuron_constrs=False,
                    )
                    output_constraints = fun_hull.cal_hull(
                        input_lower_bounds=lb, input_upper_bounds=ub
                    )
                else:
                    raise NotImplementedError(f"Method {method} is not implemented.")
                time_cal = time.perf_counter() - time_cal

                if output_constraints is None:
                    continue

                output_constraints = output_constraints.tolist()

                file.write(
                    f"{time_cal}\t{len(output_constraints)}\t{output_constraints}\t"
                    f"{lb.tolist()}\t{ub.tolist()}\n"
                )

            file.close()
            print(f"[INFO] Save to {saved_file_path}")

    print(f"[INFO] Done in {time.perf_counter() - time_total:.2f} seconds")
