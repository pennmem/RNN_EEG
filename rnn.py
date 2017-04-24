# Recurrent neural net for EEG data
# Each list consists of 12 items
# Lists are assumed to be independent
# The model
# a(t) = b + W h(t-1) + U x(t)
# h(t) = tanh(a(t))
# o(t) = c + V h(t)
# y_hat(t) = softmax(o(t))
# L(t) = -log P(y(t) | y_hat(t))



import theano
import theano.tensor as T
import numpy as np
import timeit


# logistic layer
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value = np.zeros((n_in,n_out), dtype = theano.config.floatX), name = 'W', borrow = True)
        self.b = theano.shared(value = np.zeros((1,n_out), dtype = theano.config.floatX), name = 'b', borrow = True, broadcastable = (True,False))
        self.input = input
        self.p_1 = T.nnet.softmax(T.dot(self.input,self.W) + self.b)
        self.y_pred = T.argmax(self.p_1, axis = 1) # find argmax
        self.params = [self.W, self.b]


    def negative_log_likelihood(self,y):
        # cost function
        return -T.mean(T.log(self.p_1)[T.arange(y.shape[0]),y])


    def errors(self,y):
        # negative log-likelihood
        return T.mean(T.neq(self.y_pred,y))


# Recurrent layer
class Recurrent(object):
    # n_input : input dimension (#elec x 8 frequencies)
    # n_h : dimension of hidden layer
    # rng : numpy random number generator
    # U : transforms input features into hidden-layer features
    # W : transform h(t-1) to h(t)
    # b : intercept

    def __init__(self, rng, input, n_input, n_h,h0 = None, U = None, W = None, b = None, activation = T.tanh):
        # initial state
        if h0 is None:
            h0_values = np.zeros((n_h,), dtype = theano.config.floatX)
            self.h0 = theano.shared(name = 'h0', value = h0_values)
        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-np.sqrt(3. / n_h), high=np.sqrt(3. / n_h), size=(n_input, n_h)),
                dtype=theano.config.floatX)
        if activation == T.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(name='W', value=W_values, borrow=True)

        if U is None:
            U_values = np.asarray(
                rng.uniform(low=-np.sqrt(6. / (n_input + n_h)), high=np.sqrt(6. / (n_input + n_h)), size=(n_input, n_h)),
                dtype=theano.config.floatX)
            if activation == T.nnet.sigmoid:
                U_values *= 4
        self.U = theano.shared(name='U', value=W_values, borrow=True)

        self.params[self.U,self.W,self.b]

# Recurrent neural network object
class RNN(object):
