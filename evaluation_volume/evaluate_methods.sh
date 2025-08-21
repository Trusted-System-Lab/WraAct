#!/bin/bash

cd ./polytope_samples
python3 generate_polytope_samples.py

cd ../polytope_bounds
python3 calculate_polytope_bounds.py

cd ../function_hulls
python3 calculate_function_hulls.py

cd ../hull_volumes
python3 calculate_volumes.py


