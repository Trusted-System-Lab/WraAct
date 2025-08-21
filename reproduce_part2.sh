#!/bin/bash
set -e

echo "[INFO] Entering evaluation_verification directory..."
cd evaluation_verification

echo "[INFO] Running all verification tests..."
bash run_all.sh

cd ..

echo "[INFO] Completed all verification tests, you can check the results in the evaluation_verification/logs directory."
