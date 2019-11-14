#BP w noisy activations
#python3 ./cifar10_conv_bp_noise.py --gpu 1 --name cifar10_conv_bp_noise_run01 --save 1
#python3 ./cifar10_conv_bp_noise.py --gpu 1 --name cifar10_conv_bp_noise_run02 --save 1
#python3 ./cifar10_conv_bp_noise.py --gpu 1 --name cifar10_conv_bp_noise_run03 --save 1

#DFA w noisy activations
python3 ./cifar10_conv_noise.py --gpu 2 --name cifar10_conv_dfa_noise_run01 --save 1
python3 ./cifar10_conv_noise.py --gpu 2 --name cifar10_conv_dfa_noise_run02 --save 1
python3 ./cifar10_conv_noise.py --gpu 2 --name cifar10_conv_dfa_noise_run03 --save 1

#Synthetic gradient runs
#python3 ./cifar10_conv_sg.py --gpu 3 --name cifar10_conv_sg_run01 --save 1
#python3 ./cifar10_conv_sg.py --gpu 3 --name cifar10_conv_sg_run02 --save 1
#python3 ./cifar10_conv_sg.py --gpu 3 --name cifar10_conv_sg_run03 --save 1