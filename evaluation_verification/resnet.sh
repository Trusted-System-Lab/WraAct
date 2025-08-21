#!/bin/bash

python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --log_name cifar_tanh_resnet2b_ROVER
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --log_name cifar_tanh_resnet4b_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --log_name cifar_leaky_relu_resnet2b_ROVER
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --log_name cifar_leaky_relu_resnet4b_ROVER
python3 exp.py --net_file_path ../nets/elu/cifar_elu_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --log_name cifar_elu_resnet2b_ROVER
python3 exp.py --net_file_path ../nets/elu/cifar_elu_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --log_name cifar_elu_resnet4b_ROVER


python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt lp --log_name cifar_tanh_resnet2b_ROVER_LP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt lp --log_name cifar_tanh_resnet4b_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt lp --log_name cifar_leaky_relu_resnet2b_ROVER_LP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt lp --log_name cifar_leaky_relu_resnet4b_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt lp --log_name cifar_elu_resnet2b_ROVER_LP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt lp --log_name cifar_elu_resnet4b_ROVER_LP


python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt mnlp --log_name cifar_tanh_resnet2b_ROVER_MNLP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt mnlp --log_name cifar_tanh_resnet4b_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt mnlp --log_name cifar_leaky_relu_resnet2b_ROVER_MNLP
python3 exp.py --net_file_path ../nets/leaky_relu/cifar_leaky_relu_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt mnlp --log_name cifar_leaky_relu_resnet4b_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_resnet2b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt mnlp --log_name cifar_elu_resnet2b_ROVER_MNLP
python3 exp.py --net_file_path ../nets/elu/cifar_elu_resnet4b.onnx --dataset cifar10 --perturbation_radius 0.00392156862 --bp rover --opt mnlp --log_name cifar_elu_resnet4b_ROVER_MNLP