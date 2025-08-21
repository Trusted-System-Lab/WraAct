#!/bin/bash
set -e  # Exit on error

echo "[INFO] Entering evaluation_verification directory..."
cd evaluation_verification

echo "[INFO] Running exp_test.py ..."
python3 exp_test.py

echo "[INFO] Returning to previous directory..."
cd ..

echo "[INFO] Done."
