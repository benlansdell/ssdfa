
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Linear

class FullyConnected(Layer):

    def __init__(self, input_shape, size, init, activation=None, bias=0., use_bias=False, alpha=0., name=None, load=None, train=True):

        self.input_size = input_shape
        self.output_size = size
        self.size = [self.input_size, self.output_size]

        bias = np.ones(shape=self.output_size) * bias
        self.use_bias = use_bias

        self.alpha = alpha
        self.activation = Linear() if activation == None else activation
        self.name = name
        self._train = train

        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load).item()
            weights = weight_dict[self.name]
            bias = weight_dict[self.name + '_bias']
        else:
            if init == "zero":
                weights = np.zeros(shape=self.size)
            elif init == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.input_size)
                weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.size)
            elif init == "alexnet":
                weights = np.random.normal(loc=0.0, scale=0.01, size=self.size)
            else:
                fan_in = self.input_size
                fan_out = self.output_size
                high = np.sqrt(6. / (fan_in + fan_out))
                low = -high
                weights = np.random.uniform(low=low, high=high, size=self.size)
        
        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    ###################################################################

    def forward(self, X):
        Z = tf.matmul(X, self.weights) 
        if self.use_bias:
            Z = Z + self.bias
        A = self.activation.act_forward(Z)
        return {'aout':A, 'cache':{}}
            
    def backward(self, AI, AO, DO, cache=None):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights))
        return {'dout':DI, 'cache':{}}
        
    def gv(self, AI, AO, DO, cache=None):
        if not self._train:
            return []
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
        assert(False)
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones_like(AI)
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)
        
        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
        
    def lel_backward(self, AI, AO, E, DO, Y, cache):
        return self.backward(AI, AO, DO, cache)

    def lel_gv(self, AI, AO, E, DO, Y, cache):
        return self.gv(AI, AO, DO, cache)
        
    def lel(self, AI, AO, E, DO, Y):
        return self.train(AI, AO, DO)
        
        
