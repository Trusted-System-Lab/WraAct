#!/bin/bash

python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_3_500.onnx --dataset mnist --perturbation_radius 0.02 --bp crown --opt mnlp --log_name mnist_sigmoid_3_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_6_500.onnx --dataset mnist --perturbation_radius 0.03 --bp crown --opt mnlp --log_name mnist_sigmoid_6_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_convsmall.onnx --dataset mnist --perturbation_radius 0.01 --bp crown --opt mnlp --log_name mnist_sigmoid_convsmall_CROWN_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_convmed.onnx --dataset mnist --perturbation_radius 0.065 --bp crown --opt mnlp --log_name mnist_sigmoid_convmed_CROWN_MNLP


python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_3_500.onnx --dataset mnist --perturbation_radius 0.02 --bp deeppoly --opt mnlp --log_name mnist_sigmoid_3_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_6_500.onnx --dataset mnist --perturbation_radius 0.03 --bp deeppoly --opt mnlp --log_name mnist_sigmoid_6_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_convsmall.onnx --dataset mnist --perturbation_radius 0.01 --bp deeppoly --opt mnlp --log_name mnist_sigmoid_convsmall_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/mnist_sigmoid_convmed.onnx --dataset mnist --perturbation_radius 0.065 --bp deeppoly --opt mnlp --log_name mnist_sigmoid_convmed_DEEPPOLY_MNLP


python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_3_500.onnx --dataset mnist --perturbation_radius 0.01 --bp crown --opt mnlp --log_name mnist_tanh_3_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_6_500.onnx --dataset mnist --perturbation_radius 0.025 --bp crown --opt mnlp --log_name mnist_tanh_6_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_convsmall.onnx --dataset mnist --perturbation_radius 0.01 --bp crown --opt mnlp --log_name mnist_tanh_convsmall_CROWN_MNLP
python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_convmed.onnx --dataset mnist --perturbation_radius 0.04 --bp crown --opt mnlp --log_name mnist_tanh_convmed_CROWN_MNLP


python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_3_500.onnx --dataset mnist --perturbation_radius 0.01 --bp deeppoly --opt mnlp --log_name mnist_tanh_3_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_6_500.onnx --dataset mnist --perturbation_radius 0.025 --bp deeppoly --opt mnlp --log_name mnist_tanh_6_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_convsmall.onnx --dataset mnist --perturbation_radius 0.01 --bp deeppoly --opt mnlp --log_name mnist_tanh_convsmall_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/tanh/mnist_tanh_convmed.onnx --dataset mnist --perturbation_radius 0.04 --bp deeppoly --opt mnlp --log_name mnist_tanh_convmed_DEEPPOLY_MNLP