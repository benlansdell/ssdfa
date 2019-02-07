
import tensorflow as tf
import numpy as np

###################################################################

# TODO account for sparsity
# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Memory:

    def __init__(self):
        super().__init__()

    def read(self, shape_X):
        pass

    def write(self, shape_X):
        pass

    def total(self):
        pass

###################################################################           

class DRAM(Memory):

    def __init__(self):
        self.read_count = 0
        self.write_count = 0

    def read(self, shape_X):
        self.read_count += np.prod(shape_X)

    def write(self, shape_X):
        self.write_count += np.prod(shape_X)
        
    def total(self):
        return {'read': self.read_count, 'write': self.write_count}
        
###################################################################           

class RRAM(Memory):

    def __init__(self):
        super().__init__()

    def read(self, shape_X):
        pass

    def write(self, shape_X):
        pass

    def total(self):
        pass
        
###################################################################           

