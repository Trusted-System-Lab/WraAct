#!/bin/bash

python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_3_500.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --log_name cifar_leaky_relu_3_500_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_6_500.onnx --dataset cifar10 --perturbation_radius 0.0025 --bp rover --log_name cifar_leaky_relu_6_500_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_convsmall.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --log_name cifar_leaky_relu_convsmall_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_convmed.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --log_name cifar_leaky_relu_convmed_ROVER
python3 exp.py --net_file_path ../nets/elu/cifar_elu_3_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp rover --log_name cifar_elu_3_500_ROVER
python3 exp.py --net_file_path ../nets/elu/cifar_elu_6_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp rover --log_name cifar_elu_6_500_ROVER
python3 exp.py --net_file_path ../nets/elu/cifar_elu_convsmall.onnx --dataset cifar10 --perturbation_radius 0.0035 --bp rover --log_name cifar_elu_convsmall_ROVER
python3 exp.py --net_file_path ../nets/elu/cifar_elu_convmed.onnx --dataset cifar10 --perturbation_radius 0.0035 --bp rover --log_name cifar_elu_convmed_ROVER


python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_3_500.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --opt lp --log_name cifar_leaky_relu_3_500_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_6_500.onnx --dataset cifar10 --perturbation_radius 0.0025 --bp rover --opt lp --log_name cifar_leaky_relu_6_500_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_convsmall.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --opt lp --log_name cifar_leaky_relu_convsmall_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_convmed.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --opt lp --log_name cifar_leaky_relu_convmed_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_3_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp rover --opt lp --log_name cifar_elu_3_500_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_6_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp rover --opt lp --log_name cifar_elu_6_500_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_convsmall.onnx --dataset cifar10 --perturbation_radius 0.0035 --bp rover --opt lp --log_name cifar_elu_convsmall_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_convmed.onnx --dataset cifar10 --perturbation_radius 0.0035 --bp rover --opt lp --log_name cifar_elu_convmed_ROVER_LP


python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_3_500.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --opt mnlp --log_name cifar_leaky_relu_3_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_6_500.onnx --dataset cifar10 --perturbation_radius 0.0025 --bp rover --opt mnlp --log_name cifar_leaky_relu_6_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_convsmall.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --opt mnlp --log_name cifar_leaky_relu_convsmall_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_convmed.onnx --dataset cifar10 --perturbation_radius 0.004 --bp rover --opt mnlp --log_name cifar_leaky_relu_convmed_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_3_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp rover --opt mnlp --log_name cifar_elu_3_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_6_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp rover --opt mnlp --log_name cifar_elu_6_500_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_convsmall.onnx --dataset cifar10 --perturbation_radius 0.0035 --bp rover --opt mnlp --log_name cifar_elu_convsmall_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_convmed.onnx --dataset cifar10 --perturbation_radius 0.0035 --bp rover --opt mnlp --log_name cifar_elu_convmed_ROVER_MNLP