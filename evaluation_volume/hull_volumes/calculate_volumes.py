"""This is for calculating the volume given function hulls."""

import os
import time
from typing import List, Optional, Tuple

import numpy as np


class SamplePointsGenerator:
    def __init__(self, lower_bounds: List[float], upper_bounds: List[float]):
        assert len(lower_bounds) == len(
            upper_bounds
        ), "The length of lower bounds and upper bounds must be the same."
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def generate(
        self, points_num: int, samples_area_shape: str
    ) -> Optional[np.ndarray]:
        assert samples_area_shape in {
            "box",
            "triangle",
            "box_sigmoid",
            "box_tanh",
            "box_maxpool",
            "box_leakyrelu",
            "box_elu",
        }, f"Shape {samples_area_shape} is not supported."
        if points_num == 0:
            return None
        # print("Generate sample points...", end="")

        if samples_area_shape in [
            "box",
            "box_sigmoid",
            "box_tanh",
            "box_leakyrelu",
            "box_elu",
        ]:
            if samples_area_shape == "box_sigmoid":
                f = lambda x: 1.0 / (1.0 + np.exp(-x))
            elif samples_area_shape == "box_tanh":
                f = lambda x: np.tanh(x)
            elif samples_area_shape == "box_leakyrelu":
                f = lambda x: np.maximum(0.01 * x, x)
            elif samples_area_shape == "box_elu":
                f = lambda x: np.where(x >= 0, x, np.exp(x) - 1)
            else:
                f = lambda x: x
            lb = self.lower_bounds + [f(x) for x in self.lower_bounds]
            ub = self.upper_bounds + [f(x) for x in self.upper_bounds]
            # lb = self.lower_bounds * 2
            # ub = self.upper_bounds * 2
            sample_points = self._generate_random_points_in_box(points_num, lb, ub)

        elif samples_area_shape == "box_maxpool":
            lb = self.lower_bounds + [max(self.lower_bounds)]
            ub = self.upper_bounds + [max(self.upper_bounds)]
            sample_points = self._generate_random_points_in_box(points_num, lb, ub)

        else:
            sample_points = self._generate_random_points_in_triangle(
                points_num, self.lower_bounds, self.upper_bounds
            )

        # print(f"{time.time() - start:.4f}s")
        return sample_points

    @staticmethod
    def _generate_random_points_in_box(
        points_num: int, lower_bounds: List[float], upper_bounds: List[float]
    ) -> np.ndarray:
        vars_num = len(lower_bounds)
        r = np.random.random(
            (points_num, vars_num),
        )
        lbs = np.array(lower_bounds)
        ubs = np.array(upper_bounds)
        r = lbs + r * (ubs - lbs)

        return r

    @staticmethod
    def _generate_random_points_in_triangle(
        points_num: int, lower_bounds: List[float], upper_bounds: List[float]
    ) -> np.ndarray:
        vars_num = len(lower_bounds)
        vector_a = np.tile(
            np.hstack((np.array([lower_bounds]), np.zeros((1, vars_num)))),
            (points_num, 1),
        )
        vector_b = np.tile(np.array([upper_bounds]), (points_num, 2))
        t1 = np.random.random((points_num, vars_num))
        t2 = np.random.random((points_num, vars_num))
        bad_para_locations = np.where(t1 + t2 > 1)
        t1[bad_para_locations] = 1 - t1[bad_para_locations]
        t2[bad_para_locations] = 1 - t2[bad_para_locations]
        t1 = np.tile(t1, (1, 2))
        t2 = np.tile(t2, (1, 2))
        return vector_a * t1 + vector_b * t2


class VolumeEstimator:
    def __init__(
        self,
        constraints: np.ndarray,
        lower_bounds: List[float],
        upper_bounds: List[float],
    ):
        self.constraints = constraints
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.random_points = None

    def estimate(
        self,
        estimate_method,
        random_points: Optional[np.ndarray] = None,
        step_length=0.1,
        points_num: Optional[int] = None,
        lower_bounds: Optional[List] = None,
        upper_bounds: Optional[List] = None,
        max_points_in: int = None,
        sample_area_shape: str = None,
    ) -> Tuple[int, int]:
        if estimate_method == "random":
            points_in, points_num = self._estimate_by_random_points(
                random_points,
                points_num,
                lower_bounds,
                upper_bounds,
                sample_area_shape,
                max_points_in,
            )
        elif estimate_method == "grid":
            points_in, points_num = self._estimate_by_grid(step_length)
        else:
            raise ValueError("Unknown estimation method")
        return points_in, points_num

    def _estimate_by_random_points(
        self,
        points: Optional[np.ndarray] = None,
        points_num: Optional[int] = None,
        lower_bounds: Optional[List] = None,
        upper_bounds: Optional[List] = None,
        sample_area_shape: str = None,
        max_points_in: int = None,
        tol=1e-8,
    ) -> Tuple[int, int]:
        constraints = self.constraints

        sample_points_generator = None
        if points is not None:
            points_num = points.shape[0]
        elif max_points_in is not None:
            points_num = int(1e6)

        if points is None:
            sample_points_generator = SamplePointsGenerator(lower_bounds, upper_bounds)

        points_in = 0
        points_num2 = 0
        step_length = 10000
        for start in range(0, points_num, step_length):
            end = min(points_num, start + step_length)
            if sample_points_generator is not None:
                p = sample_points_generator.generate(end - start, sample_area_shape)
            else:
                p = points[start:end]
            points_num2 += p.shape[0]
            ax = constraints[:, 1:] @ p.T + constraints[:, :1]
            points_in += np.count_nonzero(np.all(ax > -tol, axis=0))
            if max_points_in is not None and points_in > max_points_in:
                break

        return points_in, points_num2

    def _estimate_by_grid(self, step_length=0.1, tol=1e-8) -> Tuple[int, int]:
        constraints = self.constraints

        lower_bounds = self.lower_bounds * 2
        upper_bounds = self.upper_bounds * 2
        assert len(lower_bounds) == len(
            upper_bounds
        ), "Lower and upper bounds must have the same length"
        assert len(lower_bounds) == constraints.shape[1] - 1, (
            f"Lower and upper bounds must have the same length "
            f"as the number of constraints, "
            f"but got {len(lower_bounds)} and {len(upper_bounds)} "
            f"for lower and upper bounds, "
            f"and {constraints.shape[1] - 1} for constraints"
        )

        grid_points = (
            np.mgrid[
                tuple(
                    slice(lower, upper + step_length, step_length)
                    for lower, upper in zip(lower_bounds, upper_bounds)
                )
            ]
            .reshape(len(lower_bounds), -1)
            .T
        )

        ax = constraints[:, 1:] @ grid_points.T + constraints[:, :1]

        points_in = np.count_nonzero(np.all(ax >= 0, axis=0))
        points_num = grid_points.shape[0]

        return points_in, points_num


if __name__ == "__main__":
    time_start = time.perf_counter()

    print(f"[INFO] Start calculating volumes...")

    # Get the current directory and go to the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_constraints_dir = os.path.join(current_dir, "../function_hulls")
    output_constraints_files = os.listdir(output_constraints_dir)
    output_constraints_files = [
        file
        for file in output_constraints_files
        if file.endswith(".txt") and file.startswith("polytope")
    ]

    # Sort files by dimension
    output_constraints_files.sort(
        key=lambda x: int(x.split(".")[-2].split("_")[1][:-1])
    )

    max_points_dict = {2: int(1e6), 3: int(1e6), 4: int(1e6)}

    for output_constraints_file in output_constraints_files:
        args = output_constraints_file.split(".")[0].split("_")
        dim = int(args[1].replace("d", ""))
        if dim > 4:
            # We only consider the dimensions up to 4
            continue

        print(f"[INFO] Process {output_constraints_file}...")

        method = args[4]
        max_points_in = max_points_dict[dim]

        with open(
            os.path.join(output_constraints_dir, output_constraints_file), "r"
        ) as f:
            lines = f.readlines()

        saved_file_path = output_constraints_file.replace(".txt", "_volume.txt")
        if "sigmoid" in method:
            sample_area_shape = "box_sigmoid"
        elif "tanh" in method:
            sample_area_shape = "box_tanh"
        elif "maxpool" in method:
            sample_area_shape = "box_maxpool"
        elif "leakyrelu" in method in method:
            sample_area_shape = "box_leakyrelu"
        elif "elu" in method:
            sample_area_shape = "box_elu"
        else:
            sample_area_shape = "box"

        file = open(saved_file_path, "w")
        for i, line in enumerate(lines):
            line.replace("\n", "")
            line = line.split("\t")
            time_cal = float(line[0])

            constraints_num = int(line[-4])
            output_constraints = np.asarray(eval(line[-3]))
            lb = eval(line[-2])
            ub = eval(line[-1])

            volume_estimator = VolumeEstimator(output_constraints, lb, ub)
            points_in_num, points_num = volume_estimator.estimate(
                "random",
                lower_bounds=lb,
                upper_bounds=ub,
                sample_area_shape=sample_area_shape,
                max_points_in=max_points_in,
            )
            volume = points_in_num  # / points_num
            file.write(f"{time_cal}\t{constraints_num}\t{volume}\t{points_num}\n")
        file.close()
        print(f"[INFO] Results saved to {saved_file_path}")

    time_end = time.perf_counter()
    print(f"[INFO] Done in {time_end - time_start:.4f} seconds.")
