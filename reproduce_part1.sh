#!/bin/bash
set -e

echo "[INFO] Entering evaluation_volume directory..."
cd evaluation_volume

echo "[INFO] Calculating function hull volumes..."
bash evaluate_methods.sh

echo "[INFO] Generating output tables or figures..."
bash output_data.sh

cd ..

echo "[INFO] Completed evaluation_volume tasks, you can check the results in the evaluation_volume directory."
