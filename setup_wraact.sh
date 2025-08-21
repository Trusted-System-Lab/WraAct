#!/bin/bash

# === setup_wraact_env.sh ===
# One-click script to set up the Conda environment for the WraAct project

set -e  # Exit on error

# ======================================================================================
ENV_NAME="wraact"
PYTHON_VERSION="3.12"

# === Initialize conda ===
echo "[INFO] Initializing Conda..."
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    echo "[ERROR] Cannot initialize Conda. Please ensure Conda is installed correctly."
    exit 1
fi

# === Check for existing environment ===
if conda info --envs | grep -qE "^$ENV_NAME[[:space:]]"; then
    echo "[WARNING] Conda environment '$ENV_NAME' already exists."
    read -p "Do you want to delete and recreate it? [y/N]: " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        conda deactivate || true
        echo "[INFO] Removing environment '$ENV_NAME'..."
        conda env remove --name $ENV_NAME --yes
    else
        echo "[INFO] Aborting setup."
        exit 0
    fi
fi

# === Create and activate environment ===
echo "[INFO] Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION --yes

echo "[INFO] Activating environment '$ENV_NAME'..."
conda activate $ENV_NAME || {
    echo "[ERROR] Failed to activate environment. Make sure conda is properly initialized."
    exit 1
}

# ======================================================================================

# === Install PyTorch ===
echo "[INFO] Checking for CUDA..."

USE_CUDA=false
MIN_SUPPORTED_CUDA="12.1"

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")

    if [ -z "$CUDA_VERSION" ]; then
        echo "[WARNING] Cannot detect CUDA version from nvidia-smi output."
    else
        echo "[INFO] Detected CUDA version: $CUDA_VERSION"

        # Compare versions (only major.minor)
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
        REQ_MAJOR=$(echo "$MIN_SUPPORTED_CUDA" | cut -d. -f1)
        REQ_MINOR=$(echo "$MIN_SUPPORTED_CUDA" | cut -d. -f2)

        if (( CUDA_MAJOR > REQ_MAJOR )) || { (( CUDA_MAJOR == REQ_MAJOR )) && (( CUDA_MINOR >= REQ_MINOR )); }; then
            USE_CUDA=true
        else
            echo "[WARNING] Your CUDA version ($CUDA_VERSION) is too old for cu121. Will install CPU version."
        fi
    fi
else
    echo "[INFO] No NVIDIA GPU detected (nvidia-smi not found)."
fi

if [ "$USE_CUDA" = true ]; then
    echo "[INFO] Installing PyTorch with CUDA support (cu121)..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    TORCH_BACKEND="CUDA"
else
    echo "[WARNING] Installing PyTorch with CPU support..."
    echo "[WARNING] If you have a GPU, please install the GPU version manually from PyTorch official website."
    echo "[WARNING] Refer to https://pytorch.org/get-started/locally/ for more details."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    TORCH_BACKEND="CPU"

fi


# === Install other dependencies ===
echo "[INFO] Installing additional Python packages..."
pip install \
    gurobipy \
    scipy \
    onnx==1.18.0 \
    onnxruntime \
    numba \
    pycddlib==2.1.7 \
    matplotlib

# === Save requirements ===
echo "[INFO] Saving installed packages to requirements.txt"
pip freeze > requirements.txt

# === Show summary ===
echo ""
echo "[SUCCESS] Conda environment '$ENV_NAME' created and ready with PyTorch ($TORCH_BACKEND)."

# ======================================================================================
# Script to download and extract resources from Google Drive folders

# === Function: download and extract zip files ===
download_and_extract() {
    FOLDER_NAME=$1
    GDRIVE_URL=$2
    TARGET_DIR=$3

    echo ""
    echo "[INFO] Preparing to download '$FOLDER_NAME' into '$TARGET_DIR'..."

    if [ -d "$TARGET_DIR" ]; then
        echo "[WARNING] Target directory '$TARGET_DIR' already exists. It will be deleted and replaced."
        rm -rf "$TARGET_DIR"
    fi

    mkdir -p "$TARGET_DIR"

    echo "[INFO] Downloading '$FOLDER_NAME' from Google Drive..."
    gdown --folder "$GDRIVE_URL" -O "$TARGET_DIR" || {
        echo "[ERROR] Failed to download '$FOLDER_NAME'."
        exit 1
    }

    echo "[INFO] Extracting .zip files in '$TARGET_DIR'..."
    find "$TARGET_DIR" -name "*.zip" -exec unzip -o -d "$TARGET_DIR" {} \;

    echo "[INFO] Removing .zip files from '$TARGET_DIR'..."
    find "$TARGET_DIR" -name "*.zip" -delete

    echo "[SUCCESS] '$FOLDER_NAME' is downloaded and ready in $TARGET_DIR"
}

# === Check and install gdown ===
if ! command -v gdown &> /dev/null; then
    echo "[INFO] gdown not found. Installing..."
    pip install gdown
else
    echo "[INFO] gdown is already installed."
fi

# === Download folders ===
download_and_extract \
    "nets" \
    "https://drive.google.com/drive/folders/1ga8aQm2jPXHQ4BZU797UVZsE8YUoZJnS?usp=drive_link" \
    "./nets"

download_and_extract \
    "archived_logs" \
    "https://drive.google.com/drive/folders/1ZBUSzI1lMM36GxFie7dF50gihtVjZGtB?usp=drive_link" \
    "./archived_logs"

# ======================================================================================
echo "[INFO] Running minimal Gurobi model to detect license..."

# Run a minimal Gurobi model to check for license issues, redirecting output to a temporary log file
python -u <<EOF > .gurobi_license_log.tmp 2>&1
import gurobipy as gp
model = gp.Model()
x = model.addVar()
model.setObjective(x, gp.GRB.MAXIMIZE)
model.optimize()
EOF

# Check the log file for license-related messages
LICENSE_INFO=$(grep -Ei 'license|Academic|restricted|expired|ERROR' .gurobi_license_log.tmp || true)
# Clean up the temporary log file
rm -f .gurobi_license_log.tmp

if [[ -z "$LICENSE_INFO" ]]; then
    echo "[WARNING] No license info detected. Gurobi may still be usable for small models."
else
    echo "[INFO] Detected license info:"
    echo "$LICENSE_INFO"
fi

IF_RESTRICTED_GUROBI=true
# If "Restricted license" in the output, prompt the user to set up a Gurobi license
if echo "$LICENSE_INFO" | grep -q "Restricted license"; then
    echo "[WARNING] You have a restricted Gurobi license. Please set up your Gurobi license by following the instructions at Gurobi official website."
    echo "[WARNING] Refer to http://www.gurobi.com/academia/academic-program-and-licenses/ for more details."
else
    IF_RESTRICTED_GUROBI=false
    echo "[INFO] Gurobi license appears to be valid."
fi

# ======================================================================================
conda activate $ENV_NAME


echo ""
echo "[INFO] Result Summary:"
echo "[INFO] You have created a Conda environment named '$ENV_NAME' with Python $PYTHON_VERSION."
echo "[INFO] You should activate the environment using 'conda activate $ENV_NAME'."
echo "[INFO] The environment has PyTorch installed with $TORCH_BACKEND support."
if [ "$USE_CUDA" = true ]; then
    echo "[INFO] CUDA support is enabled."
else
    echo "[WARNING] CUDA support is not enabled. Please try to manually install GPU versioned at PyTorch official website. Otherwise, the computation will take much longer."
    echo "[WARNING] Refer to https://pytorch.org/get-started/locally/ for more details."
fi
echo "[INFO] You have installed the following packages:"
pip freeze | grep -E '^(torch|torchvision|torchaudio|gurobipy|scipy|onnx|onnxruntime|numba|pycddlib|matplotlib)'
if [ "$IF_RESTRICTED_GUROBI" = false ]; then
    echo "[WARNING] You have a restricted Gurobi license. Please apply your Gurobi license at Gurobi official website. Otherwise, you can not run the whole WraAct with large-scale models."
    echo "[WARNING] Refer to http://www.gurobi.com/academia/academic-program-and-licenses/ for more details."
else
    echo "[INFO] Your Gurobi license appears to be valid."
fi
echo "[INFO] You have downloaded and extracted the following resources:"
echo "[INFO] - nets: ./nets"
echo "[INFO] - archived_logs: ./archived_logs"


# ======================================================================================