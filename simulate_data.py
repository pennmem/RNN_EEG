
import theano
import theano.tensor as T
import numpy as np
import urllib
import gzip
import os
import pickle
import timeit
import scipy.io as sio
import itertools
from sklearn import metrics


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Simulate data for Recurrent Neural Network
n_train = 2700 # 2700 observations
n_in =1384
n_hid = 100
n_out = 2
h_0 = np.zeros(n_hid,)

batch_size = 12
n_batch_size = n_train/batch_size # number of lists

U = np.random.normal(0,0.04, size = (n_in,n_hid))
W = np.random.normal(0,0.02,size = (n_hid, n_hid))
b = np.zeros((n_hid,))
c = np.random.normal(0,0.2, (n_out),)
V =  np.random.normal(0,0.02,size = (n_hid,n_out))


# Training set
y_train = np.zeros((n_train,))
x_train = np.random.normal(0,1, size = (n_train,n_in))

for batch in range(n_batch_size):
    for t in range(batch_size):
        index = t + batch*batch_size
        if t == 0:
            h_tm1 = h_0
            h_t = np.tanh(np.dot(x_train[index,:],U) + np.dot(h_tm1,W) + b)
            o_t = np.dot(h_t,V) + c
            y_hat_t = softmax(o_t)
            y_train[index] = np.argmax(y_hat_t)
            h_tm1 = h_t

        else:
            h_t = np.tanh(np.dot(x_train[index,:],U) + np.dot(h_tm1,W) + b)
            o_t = np.dot(h_t, V) + c
            y_hat_t = softmax(o_t)
            y_train[index] = np.argmax(y_hat_t)
            h_tm1 = h_t




n_valid = 600 # 300 observations

# Training set
y_valid = np.zeros((n_valid,))
x_valid = np.random.normal(0,1, size = (n_valid,n_in))
n_batch_size = n_valid/batch_size

for batch in range(n_batch_size):
    for t in range(batch_size):
        index = t + batch*batch_size
        if t == 0:

            h_tm1 = h_0
            h_t = np.tanh(np.dot(x_valid[index,:],U) + np.dot(h_0,W) + b)
            o_t = np.dot(h_t,V) + c
            y_hat_t = softmax(o_t)
            y_valid[index] = np.argmax(y_hat_t)
            h_tm1 = h_t
        else:
            h_t = np.tanh(np.dot(x_valid[index,:],U) + np.dot(h_t,W) + b)
            o_t = np.dot(h_t, V) + c
            y_hat_t = softmax(o_t)
            y_valid[index] = np.argmax(y_hat_t)
            h_tm1 = h_t





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
import scipy.io as sio
import itertools
from sklearn import metrics



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
                rng.uniform(low=-np.sqrt(3. / n_h), high=np.sqrt(3. / n_h), size=(n_h, n_h)),
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
        U = theano.shared(name='U', value=U_values, borrow=True)

        if b is None:
            b_values = np.zeros((n_h,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.U = U
        self.b = b

        def recurrence(x_t, h_tm1):
            h_t = activation(T.dot(x_t, self.U) + T.dot(h_tm1, self.W) + self.b)
            return h_t

        h,updates = theano.scan(fn = recurrence, sequences = input,
                         outputs_info = [dict(initial = self.h0)], n_steps = input.shape[0])

        theano.printing.Print('print h')(h)

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
        self.predict = self.logRegressionLayer.p_1



# test mlp
# load MNIST data

# url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
# dataset = os.getcwd() + '/dataset'
#
# urllib.urlretrieve(url, dataset)
#
# with gzip.open(dataset, 'rb') as f:
#         train_set, valid_set, test_set = pickle.load(f)
# read data


train_set_x, train_set_y = x_train, y_train
valid_set_x, valid_set_y = x_valid, y_valid


#test_set_x, test_set_y = test_set
batch_size = 12


# # hyper-parameters
# learning_rate = 0.01
# L1_reg = 0
# L2_reg = 7.2e-4*2


def shared_dataset(data_xy, borrow = True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype = np.int32), borrow = borrow)
    return shared_x, shared_y


train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))
valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y))
#test_set_x, test_set_y = shared_dataset((test_set_x,test_set_y))



n_train_batches = train_set_x.get_value(borrow = True).shape[0]//batch_size
n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]//batch_size
#n_test_batches = test_set_x.get_value(borrow = True).shape[0]//batch_size

rng = np.random.RandomState()

# build model
L2_reg = 0
learning_rate = 0.01
x = T.dmatrix('x')
y = T.ivector('y')
index = T.lscalar('ind')

n_in = x_train.shape[1]
n_h = n_in
n_out = 2




rnn_classifier = RNN(rng, input = x,  n_in = n_in , n_h = n_h , n_out = 2)
cost = rnn_classifier.negative_log_likelihood(y) + rnn_classifier.L2_cost*L2_reg
gparams = [T.grad(cost, param)/batch_size for param in rnn_classifier.params]
updates = [(param, param-learning_rate*gparam) for param,gparam in zip(rnn_classifier.params, gparams)]


train_model = theano.function(inputs=[index],
                               outputs=cost,
                               updates=updates,
                               givens={
                                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: train_set_y[index * batch_size: (index + 1) * batch_size]}
                             )



validate_model = theano.function(
    inputs=[index],
    outputs=[rnn_classifier.errors(y), rnn_classifier.predict],
    givens={
        x: valid_set_x[index * batch_size:(index + 1) * batch_size],
        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
    }
)

#
# test_model = theano.function(inputs = [index], outputs = rnn_classifier.errors(y), givens = {x:test_set_x[index*batch_size:(index+1)*batch_size],
# y:test_set_y[index*batch_size:(index+1)*batch_size]})


# traning model

print '... training the model'
patience = 50000
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience//2)
best_validation_loss= np.inf
epoch = 0


start_time = timeit.default_timer()
done_looping = False
n_epochs = 1000

# implement early-stopping rule

while(epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        iter = (epoch-1)*n_train_batches + minibatch_index
        if (iter+1)%validation_frequency == 0:
            validation_losses = [validate_model(i) for i in range(n_valid_batches)]
            this_validation_loss = np.mean([loss[0] for loss in validation_losses])
            y_valid = valid_set_y.get_value()
            prob = list(itertools.chain.from_iterable([loss[1][:,1] for loss in validation_losses]))
            fpr, tpr, thresholds = metrics.roc_curve(y_valid, prob, pos_label = 1)
            auc = metrics.auc(fpr,tpr)

            #test_losses = [test_model(i) for i in range(n_test_batches)]
            #test_score = np.mean(test_losses)
            #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'%(epoch, minibatch_index+1,n_train_batches, test_score*100)

            print 'epoch %i, minibatch %i/%i, validation error %f %%'%(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
            print 'epoch %i, minibatch %i/%i, validation AUC %f %%'%(epoch, minibatch_index+1, n_train_batches, auc*100)

            #print 'epoch %i, minibatch %i/%i, validation error %f %%'.format(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss*improvement_threshold:
                    patience = max(patience, iter*patience_increase)
                    best_validation_loss = this_validation_loss

                best_validation_loss = this_validation_loss
            print "best_validation_loss: ",best_validation_loss

                #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'.format(epoch, minibatch_index+1,n_train_batches, test_score*100)
        if patience <= iter:
            done_looping = True
            break



end_time = timeit.default_timer()

print 'The code run for %d epochs, with %f epochs/sec'%(epoch, 1.*epoch/(end_time-start_time))



rnn_classifier.params[4].get_value()
rnn_classifier.params[0].get_value()
rnn_classifier.params[1].get_value()