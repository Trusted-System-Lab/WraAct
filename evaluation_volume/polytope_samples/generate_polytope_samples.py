"""This is for generating polytope samples for evaluation."""

import itertools
import random
import time
from typing import List, Optional, Tuple

import gurobipy as grb
import numpy as np
from gurobipy import GRB


class InputConstraintsGenerator:

    def __init__(self, dim: int):
        self.dim = dim

    def generate(
        self,
        method: str,
        lower_bound: float = 0,
        upper_bound: float = 0,
        constrs_num: int = 0,
    ) -> np.ndarray:
        # print("Generate Input Constraints...", end="")

        start = time.time()
        if method == "box+random":
            constriants = np.vstack(
                [
                    self.generate_random_constraints(
                        self.dim, lower_bound, upper_bound, constrs_num
                    ),
                    self.generate_box_constraints(self.dim, lower_bound, upper_bound),
                ]
            )
        elif method == "octahedron":
            constriants = self._generate_octahedron_input_constraints(self.dim)
        else:
            raise ValueError(f"Unknown method: {method}")
        # print(f"{time.time() - start:.4f}s")
        return constriants

    @staticmethod
    def _generate_octahedron_input_constraints(dim: int) -> np.ndarray:
        constraints = []
        for coeffs in itertools.product([-1, 0, 1], repeat=dim):
            if all(c == 0 for c in coeffs):
                continue
            constraint = [random.random() * 10] + [-c for c in coeffs]
            constraints.append(constraint)
        return np.asarray(constraints)

    @staticmethod
    def generate_box_constraints(
        dim: int, lower_bound: float, upper_bound: float
    ) -> np.ndarray:
        lbs, ubs = [lower_bound] * dim, [upper_bound] * dim
        lb, ub = -np.array(lbs).reshape((-1, 1)), np.array(ubs).reshape((-1, 1))
        v1, v2 = np.identity(dim), -np.identity(dim)
        return np.vstack([np.hstack([lb, v1]), np.hstack([ub, v2])])

    @staticmethod
    def generate_random_constraints(
        dim: int, lower_bound: float, upper_bound: float, number: int
    ) -> np.ndarray:
        constraints = []
        # lower_bound, upper_bound = -4., 4.
        # r = upper_bound - lower_bound
        for _ in range(number):
            # constraint = [r * random.random() + lower_bound for __ in range(dim + 1)]
            # constraint[0] = abs(constraint[0]).  # Make sure the bias is positive
            constraint = [random.random() * 5 * dim] + [
                random.random() * 2 - 1 for __ in range(dim)
            ]
            constraints.append(constraint)
        return np.asarray(constraints)


def solve_lp(
    constraints: np.ndarray, obj_func: np.ndarray, obj_type: GRB
) -> Optional[Tuple[List, float]]:
    """
    Solve linear programming problem by GUROBI.

    :param constraints: The constraints of the linear programming problem.
    :param obj_func: The objective function of the linear programming problem.
    :param obj_type: The type of the objective function, GRB.MAXIMIZE or GRB.MINIMIZE.

    :return: The optimal solution and the optimal value of the linear programming problem.
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


def _create_octahedron_approximation(constraints: np.ndarray) -> np.ndarray:
    dim = constraints.shape[1] - 1
    oct_constrs = []
    for coeffs in itertools.product([-1, 0, 1], repeat=dim):
        if all(c == 0 for c in coeffs):
            continue
        obj = np.asarray([0] + list(coeffs))
        _, bias = solve_lp(constraints, obj, GRB.MAXIMIZE)  # noqa
        constr = [bias] + [-c for c in coeffs]
        oct_constrs.append(constr)

    return np.asarray(oct_constrs)


def generate_samples(dim: int, num: int, saved_file_path: str):
    input_constrs_generator = InputConstraintsGenerator(dim)
    input_constrs_method = "box+random"
    input_constrs_box_lower_bound = -6
    input_constrs_box_upper_bound = 6
    for n in [3]:
        print(f"[INFO] Generate {dim}d polytope with {n}^{dim} constraints...")
        file_path = f"{saved_file_path[:-4]}_{n}.txt"
        file = open(file_path, "w")
        file_path_oct = None
        file_oct = None
        if dim <= 4:
            file_path_oct = f"{saved_file_path[:-4]}_{n}_oct.txt"
            file_oct = open(file_path_oct, "w")

        input_constrs_num = n**dim
        for _ in range(num):
            input_constrs = input_constrs_generator.generate(
                input_constrs_method,
                input_constrs_box_lower_bound,
                input_constrs_box_upper_bound,
                input_constrs_num,
            )

            file.write(str(input_constrs.tolist()) + "\n")
            if file_oct is not None:
                input_constrs_oct = _create_octahedron_approximation(input_constrs)
                file_oct.write(str(input_constrs_oct.tolist()) + "\n")

        file.close()
        print(f"[INFO] Saved to {file_path}")
        if file_oct is not None:
            file_oct.close()
            print(f"[INFO] Saved to {file_path_oct}")


if __name__ == "__main__":
    time_start = time.perf_counter()

    num = 30
    print(f"[INFO] Start generating polytope {num} samples...")

    for dim in range(2, 9):
        print(f"[INFO] Generate {num} {dim}d polytopes...")
        generate_samples(dim, num, f"./polytopes_{dim}d.txt")
        print(f"[INFO] Done")

    print(f"[INFO] Done in {time.perf_counter() - time_start:.2f} seconds")
