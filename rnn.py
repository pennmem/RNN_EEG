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
import urllib
import gzip
import os
import pickle
import timeit


# logistic layer
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.V = theano.shared(value = np.zeros((n_in,n_out), dtype = theano.config.floatX), name = 'V', borrow = True)
        self.c = theano.shared(value = np.zeros((1,n_out), dtype = theano.config.floatX), name = 'c', borrow = True, broadcastable = (True,False))
        self.input = input
        self.p_1 = T.nnet.softmax(T.dot(self.input,self.V) + self.c)
        self.y_pred = T.argmax(self.p_1, axis = 1) # find argmax

        # parameters in logistic layer
        self.params = [self.V, self.c]


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

    def __init__(self, rng, input, n_input, n_h, h0 = None, U = None, W = None, b = None, activation = T.tanh):
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
            W = theano.shared(name='W', value=W_values, borrow=True)

        if U is None:
            U_values = np.asarray(
                rng.uniform(low=-np.sqrt(6. / (n_input + n_h)), high=np.sqrt(6. / (n_input + n_h)), size=(n_input, n_h)),
                dtype=theano.config.floatX)
            if activation == T.nnet.sigmoid:
                U_values *= 4
                U = theano.shared(name='U', value=W_values, borrow=True)

        if b is None:
            b_values = np.zeros((n_h,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.U = U
        self.b = b

        def recurrence(x_t, h_tm1):
            h_t = activation(T.dot(x_t, self.U) + T.dot(h_tm1, self.W) + self.b)
            return h_t

        h, = theano.scan(fn = recurrence, sequences = input,
                         outputs_info = self.h0, n_steps = input.shape[0])

        print type(h)

        self.output = h
        self.params = [self.W,self.U, self.b]


# Recurrent neural network object
class RNN(object):
    def __init__(self, rng, input, n_in, n_h, n_out):
        self.recurrentLayer = Recurrent(rng = rng, input = input, n_input = n_in, n_h = n_h, activation = T.tanh)
        self.logRegressionLayer = LogisticRegression(input = self.recurrentLayer.output, n_in = n_h, n_out = n_out)
        self.L2_cost = (self.recurrentLayer.W**2).sum() + (self.recurrentLayer.U**2).sum() + (self.logRegressionLayer.V**2).sum()
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.params = self.recurrentLayer.params + self.logRegressionLayer.params
        self.input = input
        self.errors = self.logRegressionLayer.errors



# test mlp
# load MNIST data

# url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
# dataset = os.getcwd() + '/dataset'
#
# urllib.urlretrieve(url, dataset)
#
# with gzip.open(dataset, 'rb') as f:
#         train_set, valid_set, test_set = pickle.load(f)
#
# train_set_x, train_set_y = train_set
# valid_set_x, valid_set_y = valid_set
# test_set_x, test_set_y = test_set
# batch_size = 12
#
#
# # hyper-parameters
# learning_rate = 0.01
# L1_reg = 0
# L2_reg = 0.0001
#
#
# def shared_dataset(data_xy, borrow = True):
#     data_x, data_y = data_xy
#     shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
#     shared_y = theano.shared(np.asarray(data_y, dtype = np.int32), borrow = borrow)
#     return shared_x, shared_y
#
#
# train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))
# valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y))
# test_set_x, test_set_y = shared_dataset((test_set_x,test_set_y))



n_train_batches = train_set_x.get_value(borrow = True).shape[0]//batch_size
n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]//batch_size
n_test_batches = test_set_x.get_value(borrow = True).shape[0]//batch_size

rng = np.random.RandomState(1234)

# build model
L2_reg = 0.0001 # select later
learning_rate = 0.01
x = T.dmatrix('x')
y = T.ivector('y')
index = T.lscalar('ind')
n_h = 200


Recurrent(rng = rng, input = input, n_input = n_in, n_h = n_h, activation = T.tanh)

rnn_classifier = RNN(rng, input = x,  n_in = 28*28 , n_h = n_h , n_out = 10)
cost = rnn_classifier.negative_log_likelihood(y) + rnn_classifier.L2_cost*L2_reg
gparams = [T.grad(cost, param) for param in rnn_classifier.params]
updates = [(param, param-learning_rate*gparam) for param,gparam in zip(rnn_classifier.params, gparams)]


train_model = theano.function(inputs=[index],
                               outputs=cost,
                               updates=updates,
                               givens={
                                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: train_set_y[index * batch_size: (index + 1) * batch_size]}
                             )
