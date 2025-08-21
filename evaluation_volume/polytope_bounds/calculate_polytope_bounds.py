"""This code calculates the bounds of variables in polytopes using GUROBI."""

import os
import time
from typing import Optional, Tuple, List

import gurobipy as grb
import numpy as np
from gurobipy import GRB


def solve_lp(
    constraints: np.ndarray, obj_func: np.ndarray, obj_type: GRB
) -> Optional[Tuple[List, float]]:
    """
    Solve linear programming problem by GUROBI.

    :param constraints: The constraints of the linear programming problem.
    :param obj_func: The objective function of the linear programming problem.
    :param obj_type: The type of the objective function, GRB.MAXIMIZE or GRB.MINIMIZE.

    :return: The optimal solution and the optimal value of the linear programming
        problem.
    """
    model = grb.Model("Solve LP by GUROBI")
    model.setParam("OutputFlag", False)
    model.setParam("LogToConsole", 0)
    model.setParam("Method", 0)  # Simplex method
    vars_num = constraints.shape[1] - 1
    # The default ub is GRB.INFINITY and the default lb is 0, here change the lb.
    x = np.asarray(
        [1] + [model.addVar(lb=-GRB.INFINITY) for _ in range(vars_num)]
    ).reshape((vars_num + 1, 1))

    for constraint in constraints:
        model.addConstr(grb.LinExpr(np.dot(constraint, x)[0]) >= 0)

    model.setObjective(grb.LinExpr(np.dot(obj_func, x)[0]), obj_type)  # noqa
    model.optimize()
    return (model.x, model.objVal) if model.status == GRB.OPTIMAL else None


def get_bounds_of_variables(constraints: np.ndarray) -> Tuple[List, List]:
    upper_bounds, lower_bounds = [], []
    vars_num = constraints.shape[1] - 1
    for i in range(1, vars_num + 1):
        obj_func = np.zeros((1, vars_num + 1))
        obj_func[0, i] = 1
        _, upper_bound = solve_lp(constraints, obj_func[0], GRB.MAXIMIZE)  # noqa
        _, lower_bound = solve_lp(constraints, obj_func[0], GRB.MINIMIZE)  # noqa
        assert (
            upper_bound is not None and lower_bound is not None
        ), "The polytope is unbounded or the LPP is infeasible."
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return lower_bounds, upper_bounds


def cal_bounds(polytope_samples_file_path: str):

    with open(polytope_samples_file_path, "r") as f:
        lines = f.readlines()
    polytope_samples_file_path = (
        "./" + polytope_samples_file_path.split("/")[-1]
    ).replace(".txt", "_bounds.txt")

    output_file = open(polytope_samples_file_path, "w")

    for line in lines:
        constraints = np.asarray(eval(line))
        lower_bounds, upper_bounds = get_bounds_of_variables(constraints)
        output_file.write(f"({lower_bounds}, {upper_bounds})\n")

    output_file.close()
    print(f"[INFO] Saved to {polytope_samples_file_path}.")


if __name__ == "__main__":
    time_start = time.perf_counter()

    print("[INFO]Start calculating the bounds of each dimension in the polytopes...")

    # Get the current directory and go to the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    polytope_samples_folder = os.path.join(current_dir, "../polytope_samples")
    polytope_samples_files = os.listdir(polytope_samples_folder)
    polytope_samples_files = [
        file
        for file in polytope_samples_files
        if file.endswith(".txt") and not file.endswith("oct.txt")
    ]
    polytope_samples_files.sort(key=lambda x: int(x.split(".")[-2].split("_")[1][:-1]))

    for polytope_samples_file in polytope_samples_files:
        print(f"[INFO] Process {polytope_samples_file}...")
        polytope_samples_file_path = os.path.join(
            polytope_samples_folder, polytope_samples_file
        )
        cal_bounds(polytope_samples_file_path)
        print("[INFO] Done")

    print(f"[INFO] Done in {time.perf_counter() - time_start:.2f} seconds")
