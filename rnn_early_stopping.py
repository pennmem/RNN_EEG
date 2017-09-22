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
import pandas as pd
from sklearn import metrics
import sys
import collections
import numpy as np
from sklearn.externals import joblib
from rnn_class import*
import lasagne


args = sys.argv
#subject_index = np.int(args[1])

subject_index = 1

mount_point = '/Volumes/RHINO'  # rhino mount point



subjects = np.sort(os.listdir(mount_point+ '/scratch/tphan/frequency_selection/'))  # get all subjects
subject = subjects[subject_index]
subject_dir = mount_point + '/scratch/tphan/frequency_selection/' + subject + '/'
dataset_dir = subject_dir + subject + '-dataset_current_freqs.pkl'
bipolar_dir = subject_dir + subject + '-bp_tal_structs.pkl'

dataset = joblib.load(dataset_dir)


y_data = dataset['recalled']
x_data = dataset['pow-mat']
serialpos = dataset['serialpos']


event_sessions = dataset['sess']
n_sess = len(np.unique(event_sessions))
list_pos = dataset['list']
unique_sessions = np.unique(event_sessions)


x_data = normalize_sessions(x_data,event_sessions)


auc_session = np.zeros(n_sess)
prob_session = []
y_session = []
L2_session = []
alpha_session = []


# hyperparameters
np.random.seed(100)
#lambda_grid = np.sort(10**np.random.uniform(-7,3,20))
lambda_grid = np.sort(10**np.random.uniform(-7,3,1))
#learning_rate_grid = np.array([1.0e-3, 1.0e-4])
learning_rate_grid = np.array([1.0e-3])

auc_array = np.zeros(shape = (len(learning_rate_grid), len(lambda_grid)))


print subject

start_time = timeit.default_timer()


for ii, sess in enumerate(unique_sessions):


    ix_out = ii

    # training + validation set

    train_set_indices =np.where(event_sessions != sess)[0].astype(int)
    x_train = x_data[train_set_indices,:]
    session_train = event_sessions[train_set_indices]
    y_train = y_data[train_set_indices]
    list_train = list_pos[train_set_indices]
    serialpos_train = serialpos[train_set_indices]

    # training set

    #train_set_mask = np.array([temp in [np.min(list_train), np.max(list_train)//2,np.max(list_train)] for temp in list_train]) # leave the last two lists for validation
    train_set_mask = list_train < np.max(list_train)-2
    train_x = x_train[train_set_mask]
    train_y = y_train[train_set_mask]
    train_list = list_train[train_set_mask]
    train_serialpos = serialpos_train[train_set_mask]

    # validation set
    valid_x = x_train[~train_set_mask]
    valid_y = y_train[~train_set_mask]
    valid_list = list_train[~train_set_mask]
    valid_serialpos = serialpos_train[~train_set_mask]

    # test set
    test_set_indices =np.where(event_sessions == sess)[0].astype(int)
    x_test = x_data[test_set_indices,:]
    session_test = event_sessions[test_set_indices]
    y_test = y_data[test_set_indices]
    list_test = list_pos[test_set_indices]
    serialpos_test = serialpos[test_set_indices]


    train_set_x, train_set_y = train_x, train_y
    valid_set_x, valid_set_y = valid_x, valid_y
    test_set_x, test_set_y = x_test, y_test


    # Nested cross validation for choosing L2_reg
    n_folds = 5
    auc_folds = np.zeros(n_folds)

    # for i in np.arange(len(lambda_grid)):
    #     for j in np.arange(len(learning_rate_grid)):
    #         auc_array[j,i] = cv(x_train, y_train, list_train, np.unique(list_train), serialpos_train, learning_rate_grid[j], lambda_grid[i])
    #
    #

    #
    index = np.argmax(auc_array)
    index = np.unravel_index(index, dims = auc_array.shape, order = 'C')
    L2_reg_opt = lambda_grid[index[0]]
    learning_rate_opt = learning_rate_grid[index[1]]


    # create weights for class imbalance
    N = np.double(len(train_set_y))
    pos_weight = sum(train_set_y)/N
    neg_weight = 1.0-pos_weight
    weights = np.array([neg_weight if i == 1 else pos_weight for i in train_set_y])
    weights = weights/np.sum(weights)*N

    # weights = np.ones(shape = weights.shape)
    train_set_weights = theano.shared(np.asarray(weights, dtype = np.float32), borrow = True)

    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y))
    test_set_x, test_set_y = shared_dataset((test_set_x,test_set_y))

    train_batches_start_indices = np.where(train_serialpos == np.min(train_serialpos))[0]

    valid_batches_start_indices = np.where(valid_serialpos == np.min(valid_serialpos))[0]

    test_batches_start_indices = np.where(serialpos_test == np.min(serialpos_test))[0]

    rng = np.random.RandomState()


    # build model

    x = T.dmatrix('x')
    y = T.ivector('y')
    class_weights = T.fvector('weights')


    index1 = T.lscalar('ind')
    index2 = T.lscalar('ind')

    n_in = x_train.shape[1]
    n_h = np.min([n_in,n_in]) # all units
    n_out = 2

    n_total = n_in*n_h + n_h**2 + n_h*n_out

    rnn_classifier = RNN(rng, input = x,  n_in = n_in , n_h = n_h , n_out = 2, weights = class_weights)

    batch_size= 12.0

    cost = rnn_classifier.negative_log_likelihood(y) + rnn_classifier.L2_cost*L2_reg_opt/N
    params = rnn_classifier.params
    updates = lasagne.updates.adadelta(cost, params, learning_rate=learning_rate_opt)


    #gparams = [T.grad(cost, param) for param in rnn_classifier.params]
    #updates = [(param, param-learning_rate_opt*gparam) for param,gparam in zip(rnn_classifier.params, gparams)]

    train_model = theano.function(inputs=[index1, index2],
                                   outputs=cost,
                                   updates=updates,
                                   givens={
                                        x: train_set_x[index1:index2],
                                        y: train_set_y[index1:index2],
                                        class_weights: train_set_weights[index1:index2]}
                                 )



    validate_model = theano.function(
        inputs=[index1, index2],
        outputs=[rnn_classifier.errors(y), rnn_classifier.predict],
        givens={
            x: valid_set_x[index1:index2],
            y: valid_set_y[index1:index2]
        }
    )


    test_model = theano.function(
        inputs=[index1, index2],
        outputs=[rnn_classifier.errors(y), rnn_classifier.predict],
        givens={
            x: test_set_x[index1:index2],
            y: test_set_y[index1:index2]
        }
    )


    # test_model = theano.function(inputs = [index], outputs = rnn_classifier.errors(y), givens = {x:test_set_x[index*batch_size:(index+1)*batch_size],
    # y:test_set_y[index*batch_size:(index+1)*batch_size]})


    # traning model

    print '... training the model'


    start_time = timeit.default_timer()


    done_looping = False
    n_epochs = 5000

    validation_loss_list = []

    auc_loss_list = []

    N_train = train_x.shape[0]
    train_batches_start_indices = np.concatenate([train_batches_start_indices, np.array([N_train])])

    N_valid = valid_x.shape[0]

    valid_batches_start_indices = np.concatenate([valid_batches_start_indices, np.array([N_valid])])

    N_test = x_test.shape[0]

    test_batches_start_indices = np.concatenate([test_batches_start_indices, np.array([N_test])])

    n_train_batches = len(train_batches_start_indices)
    n_valid_batches = len(valid_batches_start_indices)
    n_test_batches = len(test_batches_start_indices)

    patience = N_train*400
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(N_train, patience//2)
    best_validation_loss= np.inf
    epoch = 0
    iter = 0
    done_looping = False


    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for i in np.arange(n_train_batches-1):
            minibatch_avg_cost \
                = train_model(train_batches_start_indices[i], train_batches_start_indices[i+1])
            iter += train_batches_start_indices[i+1] - train_batches_start_indices[i]

            #print np.mean(np.abs(update_scale)*learning_rate/np.abs(param_scale))

            if (iter)%validation_frequency == 0:
                validation_losses = [validate_model(valid_batches_start_indices[i],valid_batches_start_indices[i+1]) for i in np.arange(n_valid_batches-1)]
                this_validation_loss = np.mean([loss[0] for loss in validation_losses])
                y_valid = valid_set_y.get_value()
                prob_valid = list(itertools.chain.from_iterable([loss[1][:,1] for loss in validation_losses]))
                fpr, tpr, thresholds = metrics.roc_curve(y_valid, prob_valid, pos_label = 1)
                auc_valid = metrics.auc(fpr,tpr)

                test_losses = [test_model(test_batches_start_indices[i],test_batches_start_indices[i+1]) for i in np.arange(n_test_batches-1)]
                this_test_loss = np.mean([loss[0] for loss in test_losses])
                y_test = test_set_y.get_value()
                prob_test = list(itertools.chain.from_iterable([loss[1][:,1] for loss in test_losses]))
                fpr, tpr, thresholds = metrics.roc_curve(y_test, prob_test, pos_label = 1)
                auc_test = metrics.auc(fpr,tpr)

                print 'epoch %i, validation error %f, valid auc %f, test auc %f ' % (epoch, this_validation_loss*100, auc_valid, auc_test)

            #   print 'epoch %i, minibatch %i/%i, validation error %f %%'.format(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss*improvement_threshold:
                        patience = max(patience, iter*patience_increase)
                        print patience
                        best_validation_loss = this_validation_loss

                    best_validation_loss = this_validation_loss
                print "best_validation_loss: ",best_validation_loss

                    #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'.format(epoch, minibatch_index+1,n_train_batches, test_score*100)
            if patience <= iter:
                done_looping = True
                break

    prob_session.append(prob_test)
    y_session.append(y_test)
    auc_session[ii] = auc_test
    L2_session.append(L2_reg_opt)
    alpha_session.append(learning_rate_opt)

    end_time = timeit.default_timer()

print 'The code run for %d epochs, with %f epochs/sec'%(epoch, 1.*epoch/(end_time-start_time))

prob_session = list(itertools.chain.from_iterable([prob for prob in prob_session]))
y_session = list(itertools.chain.from_iterable([z for z in y_session]))

fpr, tpr, thresholds = metrics.roc_curve(y_session, prob_session, pos_label = 1)
auc = metrics.auc(fpr,tpr)

print auc


end_time = timeit.default_timer()

print end_time-start_time


result = collections.OrderedDict()
result['subject'] = subject
result['lambda'] = L2_session
result['alpha'] = alpha_session
result['AUC'] = auc
result['AUC_session'] = auc_session
result['lambda_grid'] = lambda_grid
result['alpha_grid'] = learning_rate_grid


result = pd.DataFrame([result])
save_dir = mount_point + '/home2/tungphan/RNN/result/' + subject + '.pkl'
joblib.dump(result, save_dir)
