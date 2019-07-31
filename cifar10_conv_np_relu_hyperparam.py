
import argparse
import os
import sys

import time
import tensorflow as tf
import keras
import math
import numpy as np
import pickle
import numpy.random as rng

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv
from lib.NodePert import NodePert

from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.Activation import Relu
from lib.Activation import Tanh
from lib.Activation import Softmax
from lib.Activation import LeakyRelu
from lib.Activation import Linear

##############################################

def set_random_hyperparameters(args, attrs, ranges, log_scale):
    params = []
    for idx, attr in enumerate(attrs):
        val = rng.rand()*(ranges[idx][1]-ranges[idx][0])+ranges[idx][0]
        if log_scale[idx]:
            val = np.power(10, val)
        setattr(args, attrs[idx], val)
        params.append(val)
    return params

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1e-4)  #feedback weights, B, learning rate
    parser.add_argument('--sigma', type=float, default=0.1)  #node pert standard deviation
    parser.add_argument('--l2', type=float, default=0.)
    parser.add_argument('--decay', type=float, default=1.)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--bias', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--dfa', type=int, default=1)
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--init', type=str, default="sqrt_fan_in")
    parser.add_argument('--opt', type=str, default="adam")
    parser.add_argument('--N', type=int, default=30)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--name', type=str, default="cifar10_conv_np")
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()
    
    if args.gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    cifar10 = tf.keras.datasets.cifar10.load_data()
    
    ##############################################
    
    EPOCHS = args.epochs
    TRAIN_EXAMPLES = 50000
    TEST_EXAMPLES = 10000
    BATCH_SIZE = args.batch_size
    
    if args.act == 'tanh':
        act = Tanh()
    elif args.act == 'relu':
        act = Relu()
    else:
        assert(False)
    
    train_fc=True
    if args.load:
        train_conv=False
    else:
        train_conv=True
    
    weights_fc=None
    weights_conv=args.load
    
    #Setup the parameters
    attrs = ['sigma', 'alpha', 'beta']
    log_scale = [True, True, True]
    ranges = [[-4, -1], [-6,-3], [-6, -3]]
    params = []
    isnan = []

    #Here we run a bunch of times for different parameters...
    for idx in range(args.N):
    
        #Choose some random parameters...
        param = set_random_hyperparameters(args, attrs, ranges, log_scale)
        params.append(param)

        #Tell me the params....
        print('Alpha, beta, sigma are: ', args.alpha, args.beta, args.sigma)

        tf.set_random_seed(0)
        tf.reset_default_graph()
        
        batch_size = tf.placeholder(tf.int32, shape=())
        dropout_rate = tf.placeholder(tf.float32, shape=())
        learning_rate = tf.placeholder(tf.float32, shape=())
        sigma = tf.placeholder(tf.float32, shape=(), name = "Sigma")
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)
        Y = tf.placeholder(tf.float32, [None, 10])
        
        l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='conv1', load=weights_conv, train=train_conv)
        l1 = MaxPool(size=[batch_size, 32, 32, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        #Add perturbation to activity to get output to train feedback weights with
        l2p = NodePert(size=[batch_size, 16, 16, 96], sigma = sigma)
        l2 = FeedbackConv(size=[batch_size, 16, 16, 96], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv1_fb')
        
        l3 = Convolution(input_sizes=[batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='conv2', load=weights_conv, train=train_conv)
        l4 = MaxPool(size=[batch_size, 16, 16, 128], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        l5p = NodePert(size=[batch_size, 8, 8, 128], sigma = sigma)
        l5 = FeedbackConv(size=[batch_size, 8, 8, 128], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv2_fb')
        
        l6 = Convolution(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 256], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='conv3', load=weights_conv, train=train_conv)
        l7 = MaxPool(size=[batch_size, 8, 8, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        l8p = NodePert(size=[batch_size, 4, 4, 256], sigma = sigma)
        l8 = FeedbackConv(size=[batch_size, 4, 4, 256], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv3_fb')
        
        l9 = ConvToFullyConnected(shape=[4, 4, 256])
        
        l10p = NodePert(size=[batch_size, 4*4*256], sigma = sigma)
        l10 = FullyConnected(size=[4*4*256, 2048], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='fc1', load=weights_fc, train=train_fc)
        l11 = Dropout(rate=dropout_rate)
        l12 = FeedbackFC(size=[4*4*256, 2048], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc1_fb')
        
        l13p = NodePert(size=[batch_size, 2048], sigma = sigma)
        l13 = FullyConnected(size=[2048, 2048], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='fc2', load=weights_fc, train=train_fc)
        l14 = Dropout(rate=dropout_rate)
        l15 = FeedbackFC(size=[2048, 2048], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc2_fb')
        
        l16 = FullyConnected(size=[2048, 10], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=args.bias, last_layer=True, name='fc3', load=weights_fc, train=train_fc)
        
        ##############################################
        
        model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16])
        model_perturbed = Model(layers=[l0, l1, l2p, l2, l3, l4, l5p, l5, l6, l7, l8p, l8, l9, l10p, l10, l11, l12, l13p, l13, l14, l15, l16])
        
        predict = model.predict(X=X)
        predict_perturbed = model_perturbed.predict(X=X)
        
        #######
        #Pairs of perturbations and feedback weights
        #feedbackpairs = [[l2p, l2], [l5p, l5], [l8p, l8], [l10p, l12], [l13p, l15]]
        
        #Test one at a time... this works, so it must be l10p, 12 pair that fails
        feedbackpairs = [[l2p, l2], [l5p, l5], [l8p, l8], [l13p, l15]]
        
        #Get noise, feedback matrices, and loss function and unperturbed loss function, to make update rule for feedback weights
        loss = tf.reduce_sum(tf.pow(tf.nn.softmax(predict) - Y, 2), 1)/2
        loss_perturbed = tf.reduce_sum(tf.pow(tf.nn.softmax(predict_perturbed) - Y, 2), 1)/2
        
        train_B = []
        E = tf.nn.softmax(predict) - Y
        for idx, (noise, feedback) in enumerate(feedbackpairs):
            print(idx, batch_size, feedback.output_size)
            xi = tf.reshape(noise.get_noise(), (batch_size, feedback.output_size))
            B = feedback.B
            lambd = tf.matmul(tf.diag(loss_perturbed - loss)/args.sigma/args.sigma, xi)
            np_error = tf.matmul(E, B) - lambd
            grad_B = tf.matmul(tf.transpose(E), np_error)
            new_B = B.assign(B - args.beta*grad_B)
            train_B.append(new_B)
        #######
        
        weights = model.get_weights()
        
        if args.opt == "adam" or args.opt == "rms" or args.opt == "decay":
            if args.dfa:
                grads_and_vars = model.dfa_gvs(X=X, Y=Y)
            else:
                grads_and_vars = model.gvs(X=X, Y=Y)
                
            if args.opt == "adam":
                train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
            elif args.opt == "rms":
                train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
            elif args.opt == "decay":
                train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
            else:
                assert(False)
        
        else:
            if args.dfa:
                train = model.dfa(X=X, Y=Y)
            else:
                train = model.train(X=X, Y=Y)
        
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        
        
        ##############################################
        
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        
        (x_train, y_train), (x_test, y_test) = cifar10
        
        x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
        y_train = keras.utils.to_categorical(y_train, 10)
        
        x_test = x_test.reshape(TEST_EXAMPLES, 32, 32, 3)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        ##############################################
        
        filename = args.name + '.results'
        f = open(filename, "w")
        f.write(filename + "\n")
        f.write("total params: " + str(model.num_params()) + "\n")
        f.close()
        
        ##############################################
        
        train_accs = []
        test_accs = []
        
        for ii in range(EPOCHS):
            if args.opt == 'decay' or args.opt == 'gd':
                decay = np.power(args.decay, ii)
                lr = args.alpha * decay
            else:
                lr = args.alpha
                
            print (ii)
            
            #############################
            
            _count = 0
            _total_correct = 0
            
        
            #The training loop... here we add something to also update the feedback weights with the node pert
            for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
                xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
                ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
                _correct, _ = sess.run([total_correct, train], feed_dict={sigma: 0.0, batch_size: BATCH_SIZE, dropout_rate: args.dropout, learning_rate: lr, X: xs, Y: ys})
                
                #Add step to update B......
                _ = sess.run([train_B], feed_dict={sigma: args.sigma, batch_size: BATCH_SIZE, dropout_rate: args.dropout, learning_rate: lr, X: xs, Y: ys})
        
                _total_correct += _correct
                _count += BATCH_SIZE
        
            train_acc = 1.0 * _total_correct / _count
            train_accs.append(train_acc)
        
            #############################
        
            _count = 0
            _total_correct = 0
        
            for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
                xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
                ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
                _correct = sess.run(total_correct, feed_dict={sigma: 0.0, batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
                
                _total_correct += _correct
                _count += BATCH_SIZE
                
            test_acc = 1.0 * _total_correct / _count
            test_accs.append(test_acc)
            
            isnan.append(None)

            #try:
            #    trainer.train()
            #except ValueError:
            #    print("Method fails to converge for these parameters")
            #    isnan[n,m] = 1

            #Save results...
            #############################
                    
            print ("train acc: %f test acc: %f" % (train_acc, test_acc))
            
            f = open(filename, "a")
            f.write("train acc: %f test acc: %f\n" % (train_acc, test_acc))
            f.close()
            
        #Save params after each run
        fn = "./cifar10_conv_np_relu_hyperparam_search.npz"
        to_save = {
            'attr': attrs,
            'params': params,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'isnan': isnan
        }
        pickle.dump(to_save, open(fn, "wb"))

##############################################

main()