#!/bin/bash

python3 exp.py --net_file_path ../nets/maxpool/mnist_conv_maxpool.onnx --dataset mnist --perturbation_radius 0.01 --bp crown --opt mnlp --log_name mnist_conv_maxpool_CROWN_MNLP
python3 exp.py --net_file_path ../nets/maxpool/mnist_conv_maxpool.onnx --dataset mnist --perturbation_radius 0.01 --bp deeppoly --opt mnlp --log_name mnist_conv_maxpool_DEEPPOLY_MNLP


python3 exp.py --net_file_path ../nets/maxpool/cifar_conv_maxpool.onnx --dataset cifar10 --perturbation_radius 0.001 --bp crown --opt mnlp --log_name cifar_conv_maxpool_CROWN_MNLP
python3 exp.py --net_file_path ../nets/maxpool/cifar_conv_maxpool.onnx --dataset cifar10 --perturbation_radius 0.001 --bp deeppoly --opt mnlp --log_name cifar_conv_maxpool_DEEPPOLY_MNLP
