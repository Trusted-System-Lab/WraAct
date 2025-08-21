#!/bin/bash

python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_3_500.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --log_name mnist_leaky_relu_3_500_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_6_500.onnx --dataset mnist --perturbation_radius 0.015 --bp rover --log_name mnist_leaky_relu_6_500_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_convsmall.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --log_name mnist_leaky_relu_convsmall_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_convmed.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --log_name mnist_leaky_relu_convmed_ROVER
python3 exp.py --net_file_path ../nets/elu/mnist_elu_3_500.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --log_name mnist_elu_3_500_ROVER
python3 exp.py --net_file_path ../nets/elu/mnist_elu_6_500.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --log_name mnist_elu_6_500_ROVER
python3 exp.py --net_file_path ../nets/elu/mnist_elu_convsmall.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --log_name mnist_elu_convsmall_ROVER
python3 exp.py --net_file_path ../nets/elu/mnist_elu_convmed.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --log_name mnist_elu_convmed_ROVER


python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_3_500.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt lp --log_name mnist_leaky_relu_3_500_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_6_500.onnx --dataset mnist --perturbation_radius 0.015 --bp rover --opt lp --log_name mnist_leaky_relu_6_500_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_convsmall.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --opt lp --log_name mnist_leaky_relu_convsmall_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_convmed.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --opt lp --log_name mnist_leaky_relu_convmed_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_3_500.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt lp --log_name mnist_elu_3_500_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_6_500.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --opt lp --log_name mnist_elu_6_500_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_convsmall.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt lp --log_name mnist_elu_convsmall_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_convmed.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt lp --log_name mnist_elu_convmed_ROVER_LP


python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_3_500.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt mnlp --log_name mnist_leaky_relu_3_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_6_500.onnx --dataset mnist --perturbation_radius 0.015 --bp rover --opt mnlp --log_name mnist_leaky_relu_6_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_convsmall.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --opt mnlp --log_name mnist_leaky_relu_convsmall_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/mnist_leaky_relu_convmed.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --opt mnlp --log_name mnist_leaky_relu_convmed_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_3_500.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt mnlp --log_name mnist_elu_3_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_6_500.onnx --dataset mnist --perturbation_radius 0.02 --bp rover --opt mnlp --log_name mnist_elu_6_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_convmed.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt mnlp --log_name mnist_elu_convmed_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/mnist_elu_convsmall.onnx --dataset mnist --perturbation_radius 0.03 --bp rover --opt mnlp --log_name mnist_elu_convsmall_ROVER_MNLP