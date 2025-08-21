#!/bin/bash

cd ./function_hulls
python3 organize_data.py
python3 plot_line_chart.py

cd ../hull_volumes
python3 output_table_data.py