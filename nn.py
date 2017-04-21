# Logistic Regression Class
# mini-batch gradient descent MBGD

import theano
import theano.tensor as T
import numpy as np
import timeit


import urllib
import urllib
import os
import gzip
import pickle
# neural network using theano

# load MNIST data

url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
dataset = os.getcwd() + '/dataset'

urllib.urlretrieve(url, dataset)

with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

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






class HiddenLayer(object):
        def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation = T.tanh):
                self.input = input
                if W is None:
                        W_values = np.asarray(rng.uniform(low = -np.sqrt(6./(n_in + n_out)), high = np.sqrt(6./(n_in + n_out)), size = (n_in, n_out)), dtype = theano.config.floatX)
                if activation == T.nnet.sigmoid:
                        W_values *=4
                W = theano.shared(value = W_values, name = 'W', borrow = True)

                if b is None:
                        b_values = np.zeros((n_out,), dtype = theano.config.floatX)
                        b = theano.shared(value=b_values, name = 'b', borrow = True)

                self.W = W
                self.b = b
                lin_output = T.dot(input, self.W) + self.b
                self.output = (lin_output if activation is None else activation(lin_output))
                self.params = [self.W, self.b]



class MLP(object):
        def __init__(self, rng, input, n_in, n_hidden, n_out):
                self.hiddenLayer = HiddenLayer(rng = rng, input = input, n_in = n_in, n_out = n_hidden, activation = T.tanh)
                self.logRegressionLayer = LogisticRegression(input = self.hiddenLayer.output, n_in = n_hidden, n_out = n_out)
                self.L1_cost = (abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum())  # L1 penalty term
                self.L2_cost = ((self.hiddenLayer.W**2).sum() + (self.logRegressionLayer.W**2).sum())  # L2 penalty term
                self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
                self.params = self.hiddenLayer.params + self.logRegressionLayer.params
                self.input = input
                self.errors = self.logRegressionLayer.errors

# test mlp
# load MNIST data

url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
dataset = os.getcwd() + '/dataset'

urllib.urlretrieve(url, dataset)

with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set
test_set_x, test_set_y = test_set
batch_size = 20


# hyper-parameters
learning_rate = 0.01
L1_reg = 0
L2_reg = 0.0001


def shared_dataset(data_xy, borrow = True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype = np.int32), borrow = borrow)
    return shared_x, shared_y


train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))
valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y))
test_set_x, test_set_y = shared_dataset((test_set_x,test_set_y))



n_train_batches = train_set_x.get_value(borrow = True).shape[0]//batch_size
n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]//batch_size
n_test_batches = test_set_x.get_value(borrow = True).shape[0]//batch_size

rng = np.random.RandomState(12234334)

x = T.dmatrix("x")
y = T.ivector("y")
index = T.lscalar('ind')

n_hidden = 500
classifier = MLP(rng = rng, input = x, n_in = 28*28, n_hidden = n_hidden, n_out = 10)

cost = (classifier.negative_log_likelihood(y) + classifier.L1_cost*L1_reg +  classifier.L2_cost*L2_reg)

test_model = theano.function(inputs = [index], outputs = classifier.errors(y), givens = {x:test_set_x[index*batch_size:(index+1)*batch_size],
y:test_set_y[index*batch_size:(index+1)*batch_size]})


validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
        x: valid_set_x[index * batch_size:(index + 1) * batch_size],
        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
    }
)

# training model
gparams = [T.grad(cost,param) for param in classifier.params]
updates = [(param, param-learning_rate*gparam) for param,gparam in zip(classifier.params, gparams)]

train_model = theano.function(inputs=[index],
                               outputs=cost,
                               updates=updates,
                               givens={
                                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: train_set_y[index * batch_size: (index + 1) * batch_size]}
                             )





# traing model

print '... training the model'
patience =10000
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
            this_validation_loss = np.mean(validation_losses)

            test_losses = [test_model(i) for i in range(n_test_batches)]
            test_score = np.mean(test_losses)
            print 'epoch %i, minibatch %i/%i, test error of best model %f %%'%(epoch, minibatch_index+1,n_train_batches, test_score*100)

            print 'epoch %i, minibatch %i/%i, validation error %f %%'%(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
            #print 'epoch %i, minibatch %i/%i, validation error %f %%'.format(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss*improvement_threshold:
                    patience = max(patience, iter*patience_increase)
                    best_validation_loss = this_validation_loss

                best_validation_loss = this_validation_loss

                #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'.format(epoch, minibatch_index+1,n_train_batches, test_score*100)
        if patience <= iter:
            done_looping = True
            break



end_time = timeit.default_timer()

print 'The code run for %d epochs, with %f epochs/sec'%(epoch, 1.*epoch/(end_time-start_time))



