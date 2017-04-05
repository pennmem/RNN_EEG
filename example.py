import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value = np.zeros(n_in,n_out), dtype = theano.config.floatX, name = 'W', borrow = True)
        

