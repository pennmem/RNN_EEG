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
import pandas as pd
from sklearn import metrics
import sys
import pandas as pd
import collections
from scipy.stats.mstats import zscore
from rnn_class import*
from theano import printing
import lasagne
import theano

def normalize_sessions(pow_mat, event_sessions):
    sessions = np.unique(event_sessions)
    for sess in sessions:
        sess_event_mask = (event_sessions == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat


def RMSprop(cost, params, lr = 0.001, rho = 0.99, epsilon = 1.0e-3):
    gparams = [T.grad(cost, param) for param in params]
    updates = []
    for param, gparam in zip(params, gparams):

        acc = theano.shared(param.get_value()*0., broadcastable = param.broadcastable)
        acc_new = rho*acc + (1.0-rho)*gparam**2
        gradient_scaling = T.sqrt(acc_new) + epsilon

        #printing.Print(gradient_scaling)

        gparam = gparam/gradient_scaling

        updates.append((acc, acc_new))
        updates.append((param, param-lr*gparam))

    return updates


# logistic layer
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, weights):
        self.V = theano.shared(value = np.zeros((n_in,n_out), dtype = theano.config.floatX), name = 'V', borrow = True)
        self.c = theano.shared(value = np.zeros((1,n_out), dtype = theano.config.floatX), name = 'c', borrow = True, broadcastable = (True,False))
        self.input = input
        self.weights = weights
        self.p_1 = T.nnet.softmax(T.dot(self.input,self.V) + self.c)
        self.y_pred = T.argmax(self.p_1, axis = 1) # find argmax

        # parameters in logistic layer
        self.params = [self.V, self.c]


    def negative_log_likelihood(self,y):
        # cost function
        return -T.mean(T.log(self.p_1)[T.arange(y.shape[0]),y]*self.weights)


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

    def __init__(self, rng, input, n_input, n_h, h0 = None, U = None, W = None, b = None, activation = T.nnet.relu):
        # initial state
        if h0 is None:
            h0_values = np.zeros((n_h,), dtype = theano.config.floatX)

        self.h0 = theano.shared(name = 'h0', value = h0_values)
        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-np.sqrt(3. / n_h), high=np.sqrt(3. / n_h), size=(n_h, n_h)),
                dtype=theano.config.floatX)
            # W_values = np.asarray(np.identity(n_h), dtype = theano.config.floatX)


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

        #theano.printing.Print('print h')(h)

        print type(h)

        self.output = h
        self.params = [self.W,self.U, self.b]


# Recurrent neural network object
class RNN(object):
    def __init__(self, rng, input, n_in, n_h, n_out, weights):
        self.recurrentLayer = Recurrent(rng = rng, input = input, n_input = n_in, n_h = n_h, activation = T.nnet.relu)
        self.logRegressionLayer = LogisticRegression(input = self.recurrentLayer.output, n_in = n_h, n_out = n_out, weights = weights)
        self.L2_cost = (self.recurrentLayer.W**2).sum() + (self.recurrentLayer.U**2).sum() + (self.logRegressionLayer.V**2).sum()
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.params = self.recurrentLayer.params + self.logRegressionLayer.params
        self.input = input
        self.errors = self.logRegressionLayer.errors
        self.predict = self.logRegressionLayer.p_1



# Helper functions
def shared_dataset(data_xy, borrow = True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype = np.int32), borrow = borrow)
    return shared_x, shared_y

# inner cross-validation and parameter tuning
# Output: dictionary of cved AUC and corresponding params
def cv(x_data, y_data, list_pos, list_unique, serialpos, learning_rate, L2_reg, n_folds = 5, n_h = 500):

    folds = np.split(list_unique,n_folds)

    auc_folds = np.zeros(n_folds)
    prob_fold = []
    y_fold = []


    for i in np.arange(len(folds)):


        fold_mask = np.array([l not in folds[i] for l in list_pos])

        train_set_x = x_data[fold_mask]
        train_set_y = y_data[fold_mask]
        serialpos_train = serialpos[fold_mask]

        valid_set_x = x_data[~fold_mask]
        valid_set_y = y_data[~fold_mask]
        serialpos_valid = serialpos[~fold_mask]


        train_batches_start_indices = np.where(serialpos_train == np.min(serialpos_train))[0]

        valid_batches_start_indices = np.where(serialpos_valid == np.min(serialpos_valid))[0]


        rng = np.random.RandomState()



        N_train = np.double(len(train_set_y))
        pos_weight = sum(train_set_y)/N_train
        neg_weight = 1.0-pos_weight
        weights = np.array([neg_weight if i == 1 else pos_weight for i in train_set_y])
        weights = weights/np.sum(weights)*N_train
        train_set_weights = theano.shared(np.asarray(weights, dtype = np.float32), borrow = True)


        N_valid = len(valid_set_y)

        x = T.dmatrix('x')
        y = T.ivector('y')
        weights = T.fvector('weights')
        index1 = T.lscalar('ind')
        index2 = T.lscalar('ind')

        n_in = train_set_x.shape[1]
        n_h = np.min([n_h,n_in])
        n_out = 2

        n_total = n_in*n_h + n_h**2 + n_h*n_out


        train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))
        valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y))


        # weights = np.ones(shape = weights.shape)

        #learning_rate = 1.0e-3
        rnn_classifier = RNN(rng, input=x,  n_in=n_in, n_h=n_h, n_out=2, weights=weights)

        batch_size= 12.0


        #decay_rate = 0.9
        #cache = decay_rate*cache + (1-decay_rate)*

        cost = rnn_classifier.negative_log_likelihood(y) + rnn_classifier.L2_cost*L2_reg/N_train
        params = rnn_classifier.params
        # updates = RMSprop(cost, rnn_classifier.params,lr = learning_rate, rho = 0.95)
        #updates = nesterov_momentum(cost, params, learning_rate=1e-4, momentum=.9)
        #updates = lasagne.updates.rmsprop(cost, params, learning_rate = 1e-4)
        updates = lasagne.updates.adadelta(cost, params, learning_rate = learning_rate)


        # gparams = [T.grad(cost, param) for param in rnn_classifier.params]
        # updates = [(param, param-learning_rate*gparam) for param,gparam in zip(rnn_classifier.params, gparams)]
        # #

        # build models

        train_model = theano.function(inputs=[index1, index2],
                                       outputs=cost,
                                       updates=updates,
                                       givens={
                                            x: train_set_x[index1:index2],
                                            y: train_set_y[index1:index2],
                                            weights: train_set_weights[index1:index2]}
                                     )



        validate_model = theano.function(
            inputs=[index1, index2],
            outputs=[rnn_classifier.errors(y), rnn_classifier.predict],
            givens={
                x: valid_set_x[index1:index2],
                y: valid_set_y[index1:index2]
            }
        )

        print 'training fold {0:1d} ...'.format(i)




        start_time = timeit.default_timer()
        done_looping = False
        n_epochs = 5000

        validation_loss_list = []
        auc_loss_list = []

        # implement early-stopping rule
        N_train = np.int(N_train)
        train_batches_start_indices = np.concatenate([train_batches_start_indices, np.array([N_train])])

        valid_batches_start_indices = np.concatenate([valid_batches_start_indices, np.array([N_valid])])

        n_train_batches = len(train_batches_start_indices)
        n_test_batches = len(valid_batches_start_indices)



        patience = N_train*200
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(N_train, patience//2)
        best_validation_loss= np.inf
        epoch = 0

        iter = 0

        done_looping = False
        print learning_rate



        while(epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for i in np.arange(n_train_batches-1):
                minibatch_avg_cost= train_model(train_batches_start_indices[i], train_batches_start_indices[i+1])
                iter += train_batches_start_indices[i+1] - train_batches_start_indices[i]

                # print np.mean( np.abs(update_scale*learning_rate)/np.abs(param_scale))


                #print np.mean(np.abs(update_scale)*learning_rate/np.abs(param_scale))




                if (iter)%validation_frequency == 0:
                    validation_losses = [validate_model(valid_batches_start_indices[i],valid_batches_start_indices[i+1]) for i in np.arange(n_test_batches-1)]
                    this_validation_loss = np.mean([loss[0] for loss in validation_losses])
                    y_valid = valid_set_y.get_value()
                    prob = list(itertools.chain.from_iterable([loss[1][:,1] for loss in validation_losses]))
                    fpr, tpr, thresholds = metrics.roc_curve(y_valid, prob, pos_label = 1)
                    auc = metrics.auc(fpr,tpr)

                    validation_loss_list.append(this_validation_loss)
                    auc_loss_list.append(auc)

                    #test_losses = [test_model(i) for i in range(n_test_batches)]
                    #test_score = np.mean(test_losses)
                    #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'%(epoch, minibatch_index+1,n_train_batches, test_score*100)

                    print 'epoch %i, minibatch %i/%i, validation error %f, auc %f %%'%(epoch, i+1, n_train_batches, this_validation_loss*100, auc)
                    #print 'epoch %i, minibatch %i/%i, validation AUC %f %%'%(epoch, minibatch_index+1, n_train_batches, auc*100)

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

        prob_fold.append(prob)
        y_fold.append(y_valid)

    prob_fold = list(itertools.chain.from_iterable([prob for prob in prob_fold]))
    y_fold = list(itertools.chain.from_iterable([z for z in y_fold]))
    fpr, tpr, thresholds = metrics.roc_curve(y_fold, prob_fold, pos_label = 1)
    auc = metrics.auc(fpr,tpr)
    return auc



