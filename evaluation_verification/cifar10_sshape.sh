#!/bin/bash

python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_3_500.onnx --dataset cifar10 --perturbation_radius 0.004 --bp crown --opt mnlp --log_name cifar_sigmoid_3_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_6_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp crown --opt mnlp --log_name cifar_sigmoid_6_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_convsmall.onnx --dataset cifar10 --perturbation_radius 0.004 --bp crown --opt mnlp --log_name cifar_sigmoid_convsmall_CROWN_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_convmed.onnx --dataset cifar10 --perturbation_radius 0.01 --bp crown --opt mnlp --log_name cifar_sigmoid_convmed_CROWN_MNLP


python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_3_500.onnx --dataset cifar10 --perturbation_radius 0.004 --bp deeppoly --opt mnlp --log_name cifar_sigmoid_3_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_6_500.onnx --dataset cifar10 --perturbation_radius 0.003 --bp deeppoly --opt mnlp --log_name cifar_sigmoid_6_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_convsmall.onnx --dataset cifar10 --perturbation_radius 0.004 --bp deeppoly --opt mnlp --log_name cifar_sigmoid_convsmall_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/sigmoid/cifar_sigmoid_convmed.onnx --dataset cifar10 --perturbation_radius 0.01 --bp deeppoly --opt mnlp --log_name cifar_sigmoid_convmed_DEEPPOLY_MNLP


python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_3_500.onnx --dataset cifar10 --perturbation_radius 0.0015 --bp crown --opt mnlp --log_name cifar_tanh_3_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_6_500.onnx --dataset cifar10 --perturbation_radius 0.0020 --bp crown --opt mnlp --log_name cifar_tanh_6_500_CROWN_MNLP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_convsmall.onnx --dataset cifar10 --perturbation_radius 0.0015 --bp crown --opt mnlp --log_name cifar_tanh_convsmall_CROWN_MNLP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_convmed.onnx --dataset cifar10 --perturbation_radius 0.0025 --bp crown --opt mnlp --log_name cifar_tanh_convmed_CROWN_MNLP


python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_3_500.onnx --dataset cifar10 --perturbation_radius 0.0015 --bp deeppoly --opt mnlp --log_name cifar_tanh_3_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_6_500.onnx --dataset cifar10 --perturbation_radius 0.0020 --bp deeppoly --opt mnlp --log_name cifar_tanh_6_500_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_convsmall.onnx --dataset cifar10 --perturbation_radius 0.0015 --bp deeppoly --opt mnlp --log_name cifar_tanh_convsmall_DEEPPOLY_MNLP
python3 exp.py --net_file_path ../nets/tanh/cifar_tanh_convmed.onnx --dataset cifar10 --perturbation_radius 0.0025 --bp deeppoly --opt mnlp --log_name cifar_tanh_convmed_DEEPPOLY_MNLP

