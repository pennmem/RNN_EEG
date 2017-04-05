# Logistic Regression Class


import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value = np.zeros(n_in,n_out), dtype = theano.config.floatX, name = 'W', borrow = True)
        self.b = theano.shared(value = np.zeros(n_out,), dtype = theano.config.floatX, name = 'b', borrow = True)
        self.p_1 = 1/(1 + T.exp(-T.dot(self.x,self.W) - self.b))
        self.y_pred = T.argmax(self.p_1, axis = 1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(selfself,y):
        return -T.mean(T.log(self.p_1))


