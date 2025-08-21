# WraAct: Convex Hull Approximation for Activation Functions

This is an artifact for the tool called **WraAct**, which over-approximates the function hulls of various activation functions (including leaky ReLU, ReLU, sigmoid, tanh, and maxpool).

This artifact accompanies the paper **Convex Hull Approximation for Activation Functions**.

> **TIP**: You can download our original logs and benchmark ONNX models from [Google Drive link](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing).

**Table of Contents**

- [Quick Installation and Test](#quick-installation-and-test)
- [Quick Reproduce Results](#quick-reproduce-results)
- [Step-by-Step Installation](#step-by-step-installation)
    - [Hardware Requirements](#hardware-requirements)
    - [Download Repository](#download-repository)
    - [Python Environment](#python-environment)
    - [Python Libraries](#python-libraries)
- [Step-by-Step Reproduce Results](#step-by-step-reproduce-results)
    - [Quick Kick-the-Tires](#quick-kick-the-tires)
    - [Evaluation: Function Hull Approximation](#evaluation-function-hull-approximation)
        - [Check Achieved Results](#check-achieved-results)
        - [Reproduce Results](#reproduce-results)
    - [Evaluation: Local Robustness Verification](#evaluation-local-robustness-verification)
        - [Check Achieved Results](#check-achieved-results-1)
        - [Reproduce Results](#reproduce-results-1)
- [Reuse the Source Code](#reuse-the-source-code)
    - [Main Code Structure](#main-code-structure)
    - [How to Extend the Code](#how-to-extend-the-code)
- [License](#license)

# Quick Installation and Test

> You need the license of [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) to run the code in this repository. [Academic Named-User License](https://www.gurobi.com/features/academic-named-user-license/?_gl=1*egge2l*_up*MQ..*_gs*MQ..&gclid=CjwKCAjw3_PCBhA2EiwAkH_j4rrGVU5Rgx73EwYA6Py26R10HVHzY9Jokty2eH26q0CS7H-HFoQWihoCWw8QAvD_BwE&gbraid=0AAAAA-OoJU4jkCtivoppVlJFFFMV6UVqM) is recommended for academic users.

If you're familiar with Python and the libraries we use, you can quickly install WraAct by following these steps:

```cmd
git clone https://github.com/MrAnonymous3642/WraAct.git WraAct
cd WraAct
bash setup_wraact.sh
bash test_wraact.sh
```

This part includes the installation of the required libraries, downloading the benchmark ONNX models, downloading the archived logs of our paper, and running a quick test to check if the tool works well.

The expected outputs of `bash test_wraact.sh` is to verify a small instance of local robustness verification, and it prints the instance to verify is `UNKNOWN`, which means our sound approach cannot decide this instance is verified or not. Another possible result is `SAT`, which means the instance is verified and the local robustness is satisfied. If you see the output like this, it means the installation is successful and the tool works well.


> **NOTE**: If you encounter any issues during the installation, please refer to the [Step-by-Step Installation](#step-by-step-installation) section below for detailed instructions.

# Quick Reproduce Results

You can check the archived logs of our paper in the `archived_logs` folder.

Or you can reproduce the results in our paper by running the following commands:
The part 1 is about the volume evaluation of convex hull approximation for activation functions, and it will take about 30 minutes to run. The part 2 is about the local robustness verification, and it will take about 2~3 days to run.

```cmd
bash reproduce_part1.sh
bash reproduce_part2.sh
```

> **NOTE**: If you want to run the evaluation code in detail, please refer to the [Step-by-Step Reproduce Results in the Paper](#step-by-step-reproduce-results-in-the-paper) section below for detailed instructions.

# Step-by-Step Installation

> **NOTE:** The following demonstrates the installation of our tool _rather than_ all the baseline approaches, e.g., [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) for [CROWN](https://arxiv.org/pdf/1811.00866), [ERAN](https://github.com/eth-sri/eran) for [DeepPoly](https://dl.acm.org/doi/pdf/10.1145/3290354) and [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704) in the evaluation of the paper. The following installation instructions are for a [Linux](https://en.wikipedia.org/wiki/Linux) system (e.g., [Ubuntu](https://ubuntu.com/)). If you want to run the code on a [Microsoft Windows](https://en.wikipedia.org/wiki/Microsoft_Windows) or [macOS](https://en.wikipedia.org/wiki/MacOS) system, you just need to set up a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment with the corresponding Python libraries. The code is pure Python and does not depend on any system-specific libraries.


Our code is pure [Python](https://www.python.org/), but you need a version >=3.10 (we are using 3.12) to support some functions in [typing](https://docs.python.org/3/library/typing.html). Also, you need to install the following libraries.

> **TIP**: The whole installation process can be completed within **30 minutes** with a 100M network connection. The most time-consuming part is the installation of [PyTorch](https://pytorch.org/).

## Hardware Requirements

The code is designed to be efficient and can run on a normal PC with a good CPU and enough memory.
> **TIP**: You can run half of the benchmarks on a very normal PC (e.g., [4th Intel i7 CPU](https://www.intel.com/content/www/us/en/ark/products/series/84979/5th-generation-intel-core-i7-processors.html) with a [GTX 1080](https://www.nvidia.com/en-my/geforce/products/10series/geforce-gtx-1080/) GPU). This is enough for the kick-the-tires experiments.

All reported CPU experiments are conducted on a workstation equipped with [20 AMD EPYC 7702P 64-Core 2.00GHz CPUs](https://www.amd.com/en/products/processors/server/epyc/7002-series.html) with 100GB of main memory. GPU experiments are conducted on a workstation with [48 AMD Ryzen Threadripper PRO 5965WX 24-Core 4.5GHz CPUs](https://www.amd.com/en/products/processors/workstations/ryzen-threadripper.html), 252GB of main memory, and one [NVIDIA RTX A6000 GPU](https://www.nvidia.com/en-au/design-visualization/rtx-a6000/) with 48GB of GPU memory.

## Download Repository

> **TIP**: Before doing the following steps, you need to download and install [Git](https://git-scm.com/downloads) if you do not have it on your machine.

First, change to the directory where you want to put the repository. Then, download the repository with the following command:

```cmd
git clone https://github.com/MrAnonymous3642/WraAct.git WraAct
```

Next, change to the `WraAct` folder:

```cmd
cd WraAct
```

## Python Environment

> **TIP**: Before installing the following libraries, make sure you have installed [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.anaconda.com/miniconda/) first.

Considering the dependencies of libraries, we list the following libraries in the order of our installation. Actually, other versions of the following libraries may also work because our code does not require special methods.

We install the tool in an [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.anaconda.com/miniconda/) environment using `pip` or `conda` (some libraries do not support `conda`).

First, activate the default conda environment with the following command:

```cmd
conda activate base
```

Then, create a new conda environment for WraAct. You can name the environment as you like; here we use `wraact` as an example. This will create a new conda environment with Python 3.12 in one command:

```cmd
conda create --name wraact python=3.12
```

Next, activate the conda environment:

```cmd
conda activate wraact
```

## Python Libraries

> **TIP**: The following libraries can use higher versions in most cases, but we list the versions we used in our experiments.

**PyTorch**

You can install a CPU/GPU version by following the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/). We use the following versions, but higher versions of PyTorch and CUDA are also fine. Note that [NumPy](https://numpy.org/) will be installed after installing [PyTorch](https://pytorch.org/).

- pytorch==2.3.1
- torchvision==0.18.1
- torchaudio==2.3.1
- pytorch-cuda=12.1

**Gurobi**

Then, you need the library for the programming solver Gurobi and install it by `pip` (not supported by `conda`).
> **TIP**: To use Gurobi, you need a license. [Find a license right for you](https://www.gurobi.com/academia/academic-program-and-licenses/).

```cmd
pip install gurobipy==11.0.3
```

**SciPy, ONNX, ONNXRuntime, Numba**

You also need the following libraries, where `scipy` is for extracting sparse constraint matrices from `gurobipy`, `onnx` is for reading ONNX format models, `onnxruntime` is for checking the correctness of ONNX models, and `numba` is for compiling some functions (about tangent lines) to speed up the code.

```cmd
pip install scipy==1.15.3 onnx==1.18.0 onnxruntime==1.22.0 numba==0.61.2
```

**CDDLib**

The following library _needs a specific version_ because there are some changes in the latest version of `pycddlib` that will cause the code to fail. We use version `2.1.7` in our experiments. `pycddlib` is for calculating the vertices of a convex polytope in this work. It is compatible with `numpy` operations.

```cmd
pip install pycddlib==2.1.7
```

**Other Libraries (Matplotlib)**

You also need to install `matplotlib` if you want to produce our figures in the paper.

You also need to install the software [TexLive](https://www.tug.org/texlive/) if you want to plot LaTeX fonts (_This is not necessary, and we have commented out the related code in `WraAct/evaluation_volume/plot_line_chart.py`_).

At this point, you have installed all required libraries.

# Step-by-Step Reproduce Results

This section describes how to run the evaluation code in the paper. The evaluation code is in the `evaluation_volume` and `evaluation_verification` folders.

## Quick Kick-the-Tires

There is an example script `exp_test.py` to run a small instance of local robustness verification. You can run this Python script by the following command `bash test.sh` in the `evaluation_verification` folder to check if the tool works well.

```cmd
cd WraAct/evaluation_verification
bash test.sh
```

## Evaluation: Function Hull Approximation

The `evaluation_volume` folder contains the code about volume evaluation of convex hull approximation for activation functions. To evaluate the convex hull approximation method in [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704), you need install [ELINA](https://github.com/eth-sri/ELINA) and put it outside the `WraAct` folder if you need test the methods in [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704) called SBLM+PDDM. If you only want to use WraAct, you do not need to install ELINA.

### Check Achieved Results

The subfolder `archieved_logs/evaluation_volume` contains the experiment logs of our paper. You need to download the archived logs of our paper from the [Google Drive link](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing).

### Reproduce Results

There are 4 steps in 4 folders for different purposes. To run the following commands, make sure you are in the corresponding subfolder:

```cmd
cd WraAct/evaluation_volume
```

**Step 1: Generate Polytope Samples**

Folder `polytope_samples`: This is for generating samples of convex polytopes. This folder will contain several `.txt` files to record the generated samples after you run the following command. This process can be completed within 10s in our machine.

```cmd
python3 generate_polytope_samples.py
```

> TIP: Because SBLM+PDDM only support octahedrons as input polytopes, we will generate octahedrons for it based on the original random input polytopes. Also, you will see SBLM+PDDM will only calculate these octahedrons in the following steps.

Folder `polytope_bounds`: This is for calculating the bounds of each dimension of the given convex polytopes. It is used for the algorithm in [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704) called SBLM+PDDM. This folder will also contain several `.txt` files to record the bounds of each dimension after you run the following command. This process can be completed within **2 minutes** on our machine.

```cmd
python3 calculate_polytope_bounds.py
```

**Step 3: Calculate Function Hulls**

Folder `output_constraints`: This is for calculating the function hull approximation of activation functions. This folder will also contain several `.txt` files to record the constraints of the convex hull approximation after you run the following command. This process can be completed within **6 minutes** on our machine.

```cmd
python3 calculate_function_hulls.py
```

> WARN: When you run `output_constraints.py`, it needs to call the functions in [ELINA](https://github.com/eth-sri/ELINA) for SBLM+PDDM in the paper [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704). If you do not install [ELINA](https://github.com/eth-sri/ELINA), you will have some warnings. But it is fine, and you can still run the code.

When you have the data files in the `output_constraints` folder, you can run the following command to organize the data and plot the line chart in our paper. Sometimes, there will be some warnings about the lost data but it will not affect the results because SBLM+PDDM does not support high-dimensional polytopes. When plotting the line chart, we have disabled the latex font in the figure to avoid some errors. This just for archiving the results in our paper. If you want to plot the latex font in the figure, you can uncomment the lines in the head of the script `plot_line_chart.py`. This process can be completed within 1s.

```cmd
python3 organize_data.py
python3 plot_line_chart.py
```

**Step 4: Calculate Volumes of Function Hulls**

Folder `hull_volumes`: This is to estimate the volume of the convex hull approximation. This folder will contain several `.txt` files to record the estimated volumes after you run the following code. This process is time-consuming and is completed in **10 minutes** on our machine.

```cmd
python3 estimate_volumes.py
```

Then, you can run the following command to output the data table in our paper. The data will be printed in the terminal.

```cmd
python3 output_data_table.py
```

**Run All Steps**

If you want to run all the steps in one command. You can run the following bash script in the `evaluation_volume` folder.

```cmd
bash evaluate_volume.sh
bash output_data.sh
```

## Evaluation: Local Robustness Verification

The `evaluation_verification` folder contains the evaluation code for local robustness verification. To run the following commands, make sure you are in the corresponding subfolder:

```cmd
cd WraAct/evaluation_verification
```

### Check Achieved Results

The folder `archived_logs/evaluation_verification` contains the logs of local robustness verification in our paper. You need to download the archived logs of our paper from the [Google Drive link](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing).

### Reproduce Results

You need to first download the benchmark ONNX models from [Google Drive link](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing). All the evaluation scripts call `exp.py` for running experiments. You can see the following bash files in the `evaluation_verification` folder:

- `mnist_sshape.sh`: for S-shape functions (sigmoid, tanh) on the MNIST dataset (Table 4 in the paper)
- `cifar10_sshape.sh`: for S-shape functions (sigmoid, tanh) on the CIFAR-10 dataset (Table 4 in the paper)
- `maxpool.sh`: for maxpool on the MNIST and CIFAR-10 datasets (Table 4 in the paper)
- `mnist_relulike.sh`: for ReLU-like functions (elu, leaky ReLU) on the MNIST dataset (Table 5 in the paper)
- `cifar10_relulike.sh`: for ReLU-like functions (elu, leaky ReLU) on the CIFAR-10 dataset (Table 5 in the paper)
- `resnet.sh`: for the ResNet benchmark on the CIFAR-10 dataset (Table 6 in the paper)

Run the above bash files, and you can collect the log files in the `evaluation_verification/logs` folder.

> **ATTENTION**: Running all the code takes a long time (total 2~3 days, details shown in the paper), so we suggest you run them one by one. You can also comment out some lines in the bash files to reduce the number of experiments.

```cmd
bash mnist_sshape.sh
bash mnist_relulike.sh
bash maxpool.sh
bash cifar10_sshape.sh
bash cifar10_relulike.sh
bash resnet.sh
```

# Reuse the Source Code

## Main Code Structure

The folder structure of this repository is as follows. We only list the main folders and files here. The source code of WraAct is in the `src/` folder, which contains the code for bound propagation, function hull approximation, linear programming, model building, and utility functions. The evaluation code is in the `evaluation_volume` and `evaluation_verification` folders.

```
WraAct/                      # Main folder of WraAct
├── .temp/                   # Auto-created temporary files (e.g., downloaded datasets)
├── archived_logs/           # Archived logs of our paper (download from Google Drive)
├── evaluation_volume/       # Code for volume evaluation of convex hull approximation
├── evaluation_verification/ # Code for local robustness verification
├── nets/                    # Benchmark ONNX models (download from Google Drive)
├── src/                     # Source code of WraAct
│   ├── boundprop/           # Code for bound propagation
│   │   ├── ineq/            # Linear inequalities bounds for propagation
│   │   │   ├── backsub/     # Symbolic back-substitution for inequalities
│   │   │   └── relaxation/  # Linear relaxation for activation functions
│   │   └── base.py          # Base classes for propagation
│   ├── funchull/            # Code for function hull approximation
│   ├── linprog/             # Code for linear programming
│   ├── model/               # Code for verification models
│   └── utils/               # Miscellaneous utility functions
└── README.md
ELINA/                       # (Optional) ELINA for SBLM+PDDM in PRIMA (for baseline comparison)
ERAN/                        # (Optional) ERAN for DeepPoly and PRIMA (for baseline comparison)
auto-LiRPA/                  # (Optional) auto_LiRPA for CROWN (for baseline comparison)
```

The archived logs of our paper are in the `archieved_logs` folder. The benchmark models are in the `nets` folder. The other folders are for baseline approaches, including ELINA for SBLM+PDDM in PRIMA, ERAN for DeepPoly and PRIMA, and auto-LiRPA for vanilla CROWN.

> **NOTE**:
> - You need to download the benchmark ONNX models from the [Google Drive link](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing) and put them in the `nets` folder.
> - The MNIST and CIFAR-10 datasets will be downloaded automatically when you run the code with PyTorch. They are saved in the `.temp` folder.

> **TIP**: The following folders are not necessary for running the code, and you can skip them if you only want to run the code in this repository.
> - Download the archived logs of our paper from the [Google Drive link](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing) and put them in the `archieved_logs` folder.
> - Install [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) for [CROWN](https://arxiv.org/pdf/1811.00866) and put it in the `auto-LiRPA` folder.
> - Install [ERAN](https://github.com/eth-sri/eran) for [DeepPoly](https://dl.acm.org/doi/pdf/10.1145/3290354) and [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704) and put it in the `ERAN` folder. Note: when you install ERAN, you need to install the [ELINA](https://github.com/eth-sri/ELINA) project first and put it in the `ELINA` folder. This may require some file structure changes in the `ERAN` folder.

## How to Extend the Code

You can refer to the source code structure in the `src/` folder. The source code is organized into several modules, including bound propagation, function hull approximation, linear programming, model building, and utility functions.

- If you want to use new dataset or new models, you can take the example `evaluation_verification/exp.py` as a reference. You possibly need to modify the model arguments and the preprocessing for the dataset.
- If you want to extend new activation functions, you need to design how to calculate their relaxation in the `src/boundprop/ineq/relaxation/` folder and define their layer classes in the `src/boundprop/ineq/` folder. The back-substitution methods commonly have no need to be modified.
- If you want to extend new linear programming methods, you can refer to the `src/linprog/` folder and implement your own linear programming methods.
- If you want to extend new function hull approximation methods, you can refer to the `src/funchull/` folder and implement your own methods.
- The folder `src/model/` contains the verification models, which define the behaviors of the whole verification process. If you need to introduce new model topological structures, you may need to modify the model classes in this folder.
- The folder `src/utils/` contains some utility functions, which are used in the other modules. You can add your own utility functions here.

# License

This project is licensed under the MIT License.

