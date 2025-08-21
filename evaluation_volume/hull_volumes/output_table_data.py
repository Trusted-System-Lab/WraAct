"""This is for organizing the data of volume to output a table for the paper."""

import time

import numpy as np

time_start = time.perf_counter()


all_data = {
    2: {3: {}, 4: {}},
    3: {3: {}, 4: {}},
    4: {3: {}, 4: {}},
}

print("[INFO] Collecting data for 2d, 3d, and 4d polytopes with 3^n constraints...")

for dim in [2, 3, 4]:  # dimension
    for n in [3]:  # constraints number indicator
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
            print(
                f"[INFO] Processing dimension: {dim}, constraints number: {n}, method: {method}"
            )
            file_path = f"./polytopes_{dim}d_{n}_{method}_volume.txt"
            # If the file does not exist, skip it

            with open(file_path, "r") as f:
                lines = f.readlines()

            if not lines:
                print(f"[WARNING] No data found for {file_path}. Skipping...")
                all_data[dim][n][method] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                continue

            lines = [line.replace("\n", "").split("\t") for line in lines]
            data = np.asarray([[float(item) for item in line] for line in lines])

            cal_time = data[:, 0]
            constraints_num = data[:, 1]
            volume = data[:, -2] / data[:, -1]

            data = np.stack([cal_time, volume, constraints_num], axis=1)
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            all_data[dim][n][method] = [data_mean.tolist(), data_std.tolist()]

print(
    f"[INFO] Data collection completed in "
    f"{time.perf_counter() - time_start:.2f} seconds."
)
print(f"[INFO] Data saved to polytopes_data_dict.txt")
print(f"[INFO] Outputting the table...")
print()


def print_table(methods):
    print()
    print()
    print(" & ".join(["Dimension"] + methods) + " \\\\")
    for item_idx, item_name in enumerate(
        ["Runtime (s)", "Estimated Volume", "Number of Constraints"]
    ):
        print(f"{item_name} \\\\")
        for dim in {2, 3, 4}:
            row_str = f"{dim} & "
            for method in methods:
                mean_data = all_data[dim][n][method][0][item_idx]
                std_data = all_data[dim][n][method][1][item_idx]
                if item_idx in {0, 1}:
                    row_str += f"{mean_data:.6f} & $\\pm$({std_data:.6f}) & "
                else:
                    row_str += f"{mean_data:>8.2f} & $\\pm${f'({std_data:.2f})':<8} & "
            row_str = row_str[:-3] + " \\\\"
            print(row_str)


n = 3
methods = ["prima_sigmoid", "our_sigmoid"]
print_table(methods)
print(f"[INFO] If you do not have ELINA, the data of PRIMA will be 0.")
methods = ["prima_tanh", "our_tanh"]
print_table(methods)
print(f"[INFO] If you do not have ELINA, the data of PRIMA will be 0.")
methods = ["prima_maxpool", "our_maxpool_dlp"]
print_table(methods)
print(f"[INFO] If you do not have ELINA, the data of PRIMA will be 0.")

methods = ["single_sigmoid", "our_sigmoid"]
print_table(methods)
methods = ["single_tanh", "our_tanh"]
print_table(methods)
methods = ["single_maxpool", "our_maxpool_dlp"]
print_table(methods)
methods = ["single_leakyrelu", "our_leakyrelu"]
print_table(methods)
methods = ["single_elu", "our_elu"]
print_table(methods)

methods = ["our_sigmoid-a", "our_sigmoid-b", "our_sigmoid"]
print_table(methods)
methods = ["our_tanh-a", "our_tanh-b", "our_tanh"]
print_table(methods)
methods = ["our_elu-a", "our_elu"]
print_table(methods)
methods = ["our_maxpool_dlp-a", "our_maxpool_dlp"]
print_table(methods)

print()
print(f"[INFO] Done in {time.perf_counter() - time_start:.2f} seconds")
