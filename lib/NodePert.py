import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
#from lib.Activation import Activation
#from lib.Activation import Sigmoid

class NodePert(Layer):

    def __init__(self, size : tuple, sigma = 0., name=None):
        self.size = size
        #self.batch_size, self.h, self.w, self.f = self.size
        self.name = name
        self.sigma = sigma
        self.num_output = np.prod(self.size[1:])
        #Add noise!
        self.xi = tf.random.normal(shape=self.size, mean=0.0, stddev=self.sigma, dtype=tf.float32)

    ###################################################################
    
    def get_weights(self):
        return []
    
    def get_feedback(self):
        return []

    def get_noise(self):
        return self.xi

    def num_params(self):
        return 0
        
    def forward(self, X):
        A = X + self.xi
        #return X + self.xi
        return {'aout':A, 'cache':{}}
                
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return DO

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return DO
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
        
    # > https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    # > https://www.ics.uci.edu/~pjsadows/notes.pdf
    # > https://deepnotes.io/softmax-crossentropy
    def lel_backward(self, AI, AO, E, DO, Y):
        shape = tf.shape(AO)
        N = shape[0]
        AO = tf.reshape(AO, (N, self.num_output))
        S = tf.matmul(AO, tf.transpose(self.B))
        # should be doing cross entropy here.
        # is this right ?
        # just adding softmax ?
        ES = tf.subtract(tf.nn.softmax(S), Y)
        DO = tf.matmul(ES, self.B)
        DO = tf.reshape(DO, self.size)
        # (* activation.gradient) and (* AI) occur in the actual layer itself.
        return DO
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################