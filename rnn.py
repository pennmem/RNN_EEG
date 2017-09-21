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
subject_index = np.int(args[1])

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




#penalty_grid = 10**np.random.uniform(-6,1,20)



x_data = normalize_sessions(x_data,event_sessions)


auc_session = np.zeros(len(unique_sessions))
prob_session = []
y_session = []
L2_session = []
alpha_session = []


# hyperparameters
np.random.seed(100)
lambda_grid = np.sort(10**np.random.uniform(-7,3,20))
learning_rate_grid = np.array([1.0e-3, 1.0e-4])




auc_array = np.zeros(shape = (len(learning_rate_grid), len(lambda_grid)))

for ii, sess in enumerate(unique_sessions):


    ix_out = ii
    train_set_indices =np.where(event_sessions != sess)[0].astype(int)
    x_train = x_data[train_set_indices,:]
    session_train = event_sessions[train_set_indices]
    y_train = y_data[train_set_indices]
    list_train = list_pos[train_set_indices]
    serialpos_train = serialpos[train_set_indices]



    valid_set_indices =np.where(event_sessions == sess)[0].astype(int)
    x_valid = x_data[valid_set_indices,:]

    session_valid = event_sessions[valid_set_indices]
    y_valid = y_data[valid_set_indices]
    list_valid = list_pos[valid_set_indices]
    serialpos_valid = serialpos[valid_set_indices]


    train_set_x, train_set_y = x_train, y_train
    valid_set_x, valid_set_y = x_valid, y_valid


    # Nested cross validation for choosing L2_reg

    n_folds = 5
    auc_folds = np.zeros(n_folds)


    start_time = timeit.default_timer()

    for i in np.arange(len(lambda_grid)):
        for j in np.arange(len(learning_rate_grid)):
            auc_array[j,i] = cv(x_train, y_train, list_train, np.unique(list_train), serialpos_train, learning_rate_grid[j], lambda_grid[i])

    end_time = timeit.default_timer()


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

    train_batches_start_indices = np.where(serialpos_train == np.min(serialpos_train))[0]

    valid_batches_start_indices = np.where(serialpos_valid == np.min(serialpos_valid))[0]

    rng = np.random.RandomState()


    # build model

    x = T.dmatrix('x')
    y = T.ivector('y')
    class_weights = T.fvector('weights')


    index1 = T.lscalar('ind')
    index2 = T.lscalar('ind')

    n_in = x_train.shape[1]
    n_h = np.min([n_in,500]) # 500 latent units
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


    # test_model = theano.function(inputs = [index], outputs = rnn_classifier.errors(y), givens = {x:test_set_x[index*batch_size:(index+1)*batch_size],
    # y:test_set_y[index*batch_size:(index+1)*batch_size]})


    # traning model

    print '... training the model'


    start_time = timeit.default_timer()


    done_looping = False
    n_epochs = 500

    validation_loss_list = []

    auc_loss_list = []

    N_train = x_train.shape[0]
    train_batches_start_indices = np.concatenate([train_batches_start_indices, np.array([N_train])])

    N_valid = x_valid.shape[0]

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

    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for i in np.arange(n_train_batches-1):
            minibatch_avg_cost \
                = train_model(train_batches_start_indices[i], train_batches_start_indices[i+1])
            iter += train_batches_start_indices[i+1] - train_batches_start_indices[i]

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

                print 'epoch %i, minibatch %i/%i, validation error %f, auc %f %%'%(epoch, i+1, n_train_batches, this_validation_loss*100, auc)

            #   print 'epoch %i, minibatch %i/%i, validation error %f %%'.format(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
            #     if this_validation_loss < best_validation_loss:
            #         if this_validation_loss < best_validation_loss*improvement_threshold:
            #             patience = max(patience, iter*patience_increase)
            #             best_validation_loss = this_validation_loss
            #
            #         best_validation_loss = this_validation_loss
            #     print "best_validation_loss: ",best_validation_loss
            #
            #         #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'.format(epoch, minibatch_index+1,n_train_batches, test_score*100)
            # if patience <= iter:
            #     done_looping = True
            #     break

    prob_session.append(prob)
    y_session.append(y_valid)
    auc_session[ii] = auc
    L2_session.append(L2_reg_opt)
    alpha_session.append(learning_rate_opt)

    end_time = timeit.default_timer()


print 'The code run for %d epochs, with %f epochs/sec'%(epoch, 1.*epoch/(end_time-start_time))

prob_session = list(itertools.chain.from_iterable([prob for prob in prob_session]))
y_session = list(itertools.chain.from_iterable([z for z in y_session]))

fpr, tpr, thresholds = metrics.roc_curve(y_session, prob_session, pos_label = 1)
auc = metrics.auc(fpr,tpr)



result = collections.OrderedDict()
result['subject'] = subject
result['lambda'] = L2_session
result['alpha'] = alpha_session
result['AUC'] = auc
result['AUC_session'] = auc_session


result = pd.DataFrame([result])
save_dir = mount_point + '/home2/tungphan/RNN/result/' + subject + '.pkl'
joblib.dump(result, save_dir)
