
import argparse
import os
import sys

import tensorflow as tf
import keras
import numpy as np

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

from lib.Activation import Relu
from lib.Activation import Tanh

def set_random_hyperparameters(args, attrs, ranges, log_scale):
    params = []
    for idx, attr in enumerate(attrs):
        val = rng.rand()*(ranges[idx][1]-ranges[idx][0])+ranges[idx][0]
        if log_scale[idx]:
            val = np.power(10, val)
        setattr(args, attrs[idx], val)
        params.append(val)
    return params

##############################################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.0)  #feedback weights, B, learning rate
    parser.add_argument('--sigma', type=float, default=0.1)  #node pert standard deviation
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--bias', type=float, default=0.)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dfa', type=int, default=0)
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--feedbacklearning', type=int, default=1)  #Whether or not to learn feedback weights
    parser.add_argument('--init', type=str, default="glorot_uniform")
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--name', type=str, default="cifar100_conv")
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()
    
    if args.gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    ##############################################
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    
    train_examples = 50000
    test_examples = 10000
    
    assert(np.shape(x_train) == (train_examples, 32, 32, 3))
    y_train = keras.utils.to_categorical(y_train, 100)
    
    assert(np.shape(x_test) == (test_examples, 32, 32, 3))
    y_test = keras.utils.to_categorical(y_test, 100)
    
    ##############################################
    
    if args.act == 'tanh':
        act = Tanh()
    elif args.act == 'relu':
        act = Relu()
    else:
        assert(False)
    
    ##############################################
    
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
    train_accs = []
    test_accs = []

    #Here we run a bunch of times for different parameters...
    for idx in range(args.N):
    
        #Choose some random parameters...
        param = set_random_hyperparameters(args, attrs, ranges, log_scale)
        params.append(param)

        if args.feedbacklearning == 0:
            args.beta = 0

        #Tell me the params....
        print('Alpha, beta, sigma are: ', args.alpha, args.beta, args.sigma)

        tf.set_random_seed(0)
        tf.reset_default_graph()
        
        batch_size = tf.placeholder(tf.int32, shape=())
        dropout_rate = tf.placeholder(tf.float32, shape=())
        lr = tf.placeholder(tf.float32, shape=())
        sigma = tf.placeholder(tf.float32, shape=(), name = "Sigma")
        
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)
        Y = tf.placeholder(tf.float32, [None, 100])
        
        l0 = Convolution(input_shape=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], init=args.init, activation=act, bias=args.bias, name='conv1')
        l1 = MaxPool(size=[batch_size, 32, 32, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        #Add perturbation to activity to get output to train feedback weights with
        l2p = NodePert(size=[batch_size, 16, 16, 96], sigma = sigma)
        l2 = FeedbackConv(size=[batch_size, 16, 16, 96], num_classes=100, sparse=args.sparse, rank=args.rank, name='conv1_fb')
        
        l3 = Convolution(input_shape=[batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], init=args.init, activation=act, bias=args.bias, name='conv2')
        l4 = MaxPool(size=[batch_size, 16, 16, 128], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        l5p = NodePert(size=[batch_size, 8, 8, 128], sigma = sigma)
        l5 = FeedbackConv(size=[batch_size, 8, 8, 128], num_classes=100, sparse=args.sparse, rank=args.rank, name='conv2_fb')
        
        l6 = Convolution(input_shape=[batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 256], init=args.init, activation=act, bias=args.bias, name='conv3')
        l7 = MaxPool(size=[batch_size, 8, 8, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        l8p = NodePert(size=[batch_size, 4, 4, 256], sigma = sigma)
        l8 = FeedbackConv(size=[batch_size, 4, 4, 256], num_classes=100, sparse=args.sparse, rank=args.rank, name='conv3_fb')
        
        l9 = ConvToFullyConnected(input_shape=[4, 4, 256])
        
        l10p = NodePert(size=[batch_size, 4*4*256], sigma = sigma)
        l10 = FullyConnected(input_shape=4*4*256, size=2048, init=args.init, activation=act, bias=args.bias, name='fc1')
        
        l11 = Dropout(rate=dropout_rate)
        l12 = FeedbackFC(size=[4*4*256, 2048], num_classes=100, sparse=args.sparse, rank=args.rank, name='fc1_fb')
        
        l13p = NodePert(size=[batch_size, 2048], sigma = sigma)
        l13 = FullyConnected(input_shape=2048, size=2048, init=args.init, activation=act, bias=args.bias, name='fc2')
        
        l14 = Dropout(rate=dropout_rate)
        l15 = FeedbackFC(size=[2048, 2048], num_classes=100, sparse=args.sparse, rank=args.rank, name='fc2_fb')
        
        l16 = FullyConnected(input_shape=2048, size=100, init=args.init, bias=args.bias, name='fc3')
        
        ##############################################
        
        model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16])
        #model_perturbed = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16])
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
        
        if args.dfa:
            grads_and_vars = model.dfa_gvs(X=X, Y=Y)
        else:
            grads_and_vars = model.gvs(X=X, Y=Y)
                
        train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
        
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        
        ##############################################
        
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        
        ##############################################
        
        filename = args.name + '.results'
        f = open(filename, "w")
        f.write(filename + "\n")
        f.write("total params: " + str(model.num_params()) + "\n")
        f.close()
        
        ##############################################
        
        train_accs = []
        test_accs = []
        
        for ii in range(args.epochs):
        
            #############################
            
            if args.opt == 'decay' or args.opt == 'gd':
                decay = np.power(args.decay, ii)
                lr = args.alpha * decay
            else:
                lr = args.alpha

            _total_correct = 0
            for jj in range(0, train_examples, args.batch_size):
                s = jj
                e = min(jj + args.batch_size, train_examples)
                b = e - s
                
                xs = x_train[s:e]
                ys = y_train[s:e]
                
                _correct, _ = sess.run([total_correct, train], feed_dict={sigma: 0.0, batch_size: b, dropout_rate: args.dropout, lr: args.alpha, X: xs, Y: ys})
        
                #Add step to update B......
                _ = sess.run([train_B], feed_dict={sigma: args.sigma, batch_size: b, dropout_rate: args.dropout, lr: args.alpha, X: xs, Y: ys})
        
                _total_correct += _correct
        
            train_acc = 1.0 * _total_correct / (train_examples - (train_examples % args.batch_size))
            train_accs.append(train_acc)
        
            #############################
        
            _total_correct = 0
            for jj in range(0, test_examples, args.batch_size):
                s = jj
                e = min(jj + args.batch_size, test_examples)
                b = e - s
                
                xs = x_test[s:e]
                ys = y_test[s:e]
                
                _correct = sess.run(total_correct, feed_dict={batch_size: b, dropout_rate: 0.0, lr: 0.0, X: xs, Y: ys})
                _total_correct += _correct
                
            test_acc = 1.0 * _total_correct / (test_examples - (test_examples % args.batch_size))
            test_accs.append(test_acc)
            
            #############################
                    
            p = "%d | train acc: %f | test acc: %f" % (ii, train_acc, test_acc)
            print (p)
            f = open(filename, "a")
            f.write(p + "\n")
            f.close()
        
        ##############################################
        
        if args.save:
            [w] = sess.run([weights], feed_dict={})
            w['train_acc'] = train_accs
            w['test_acc'] = test_accs
            np.save(args.name, w)
          
        ##############################################
    
            #Save params after each run
            fn = "./cifar100_conv_np_hyperparam_search_varalpha_dfa_%d_fblearning_%d.npz"%(args.dfa,args.feedbacklearning)
            to_save = {
                'attr': attrs,
                'params': params,
                'train_accs': train_accs,
                'test_accs': test_accs,
                'isnan': isnan
            }
            pickle.dump(to_save, open(fn, "wb"))

main()