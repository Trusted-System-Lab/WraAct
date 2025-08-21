"""This is for plot the line chart in the paper."""

# You can uncomment the following lines to use pgf for LaTeX integration
# --------------------------------------------------------------------------------------
# import matplotlib
# matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
# matplotlib.rcParams.update({"font.family": "san-serif"})
# matplotlib.rcParams["text.usetex"] = True
# --------------------------------------------------------------------------------------

import time

import matplotlib.pyplot as plt
import numpy as np

time_start = time.perf_counter()

print("[INFO] Loading data...")

with open("data.txt", "r") as file:
    data_dict = eval(file.read())

print("[INFO] Plotting...")

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(
    left=0.1, right=0.98, bottom=0.15, top=0.98, wspace=0, hspace=0
)  # Set the margins
plt.tight_layout()

indices = ["a", "b"]
methods = ["our_sigmoid", "our_tanh", "our_maxpool_dlp", "our_leakyrelu", "our_elu"]
dimensions = [2, 3, 4, 5, 6, 7, 8]
ax = plt.subplot(1, 1, 1)
ax.set_xlabel("Input Dimension", fontsize=16)
ax.set_ylabel("Time(s)", fontsize=16)
ax.set_yscale("log")
ax.set_xticks(dimensions)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.grid(which="major", axis="y", linestyle="--", alpha=0.6)


styles = ["ks-", "k^--", "ko:", "kD-.", "kx-"]
legends = ["Sigmoid", "Tanh", "MaxPool", "LeakyReLU", "ELU"]

for i in range(len(methods)):
    method = methods[i]

    data = np.array(data_dict[method]).T
    positions = np.array(dimensions) + 0.3 * (i - 0.5)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    ax.errorbar(
        dimensions, means, yerr=stds, fmt=styles[i], label=legends[i], capsize=5
    )

ax.legend(loc="upper left", ncol=1, fontsize=13)
plt.show()
fig.savefig("evaluation_function_hull_scalability.pdf", bbox_inches="tight")

print(f"[INFO] Saved to evaluation_function_hull_scalability.pdf")
print(f"[INFO] Done in {time.perf_counter() - time_start:.2f} seconds")
