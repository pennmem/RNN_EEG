# Logistic Regression Class
# mini-batch gradient descent MBGD

import theano
import theano.tensor as T
import numpy as np
import timeit
import scipy.io as sio
import itertools
from sklearn import metrics

class LogisticRegression(object):
    def __init__(self, input, n_feature, n_class, weights):
        self.W = theano.shared(value = np.zeros((n_feature,n_class), dtype = theano.config.floatX), name = 'W', borrow = True)
        self.b = theano.shared(value = np.zeros((1,n_class), dtype = theano.config.floatX), name = 'b', borrow = True, broadcastable = (True,False))
        self.input = input
        self.p_1 = T.nnet.softmax(T.dot(self.input,self.W) + self.b)
        self.y_pred = T.argmax(self.p_1, axis = 1) # find argmax
        self.L2_cost = (self.W**2).sum()
        self.weights = weights
        self.params = [self.W, self.b]



    def negative_log_likelihood(self,y):
        # cost function
        return -T.mean(T.log(self.p_1)[T.arange(y.shape[0]),y]*self.weights)


    def errors(self,y):
        # negative log-likelihood
        return T.mean(T.neq(self.y_pred,y))






def shared_dataset(data_xy, borrow = True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype = np.int32), borrow = borrow)
    return shared_x, shared_y


dir = '/Volumes/RHINO/scratch/tphan/FR1/R1065J.mat'
data = sio.loadmat(dir)
session = data['session']
pos = data['pos']
y_data = data['y']
x_data = data['x']

# x = x[1:,:]
# y = y[1:]
# session = session[:,1:]
# pos = pos[:,1:]
# session = session[:,1:]
unique_sessions = np.unique(session)

for sess in unique_sessions :
    indices = np.where(session == sess)[1]
    x_data[indices,:] = np.apply_along_axis(lambda z: (z-np.mean(z))/np.std(z) ,0, x_data[indices,:])




auc_session = np.zeros(len(unique_sessions))
prob_session = []
y_session = []
L2_grid = np.exp(np.arange(-10,10, step = 1))

n_samp =1
auc_grid = np.zeros(n_samp)
lambda_grid = np.zeros(n_samp)


for k in range(n_samp):
    L2_reg = 1.0
    prob_session = []
    y_session = []
    for ii in range(len(unique_sessions)):
        ix_out = ii
        train_set_indices =np.where(session != unique_sessions[ix_out])[1].astype(int)
        x_train = x_data[train_set_indices,:]
        session_train = session[:,train_set_indices]
        y_train = y_data[0,train_set_indices]
        N = np.double(len(y_train))
        pos_weight = sum(y_train)/N

        neg_weight = 1.0-pos_weight
        weights = np.array([neg_weight if i == 1 else pos_weight for i in y_train])
        weights = weights/np.sum(weights)*N

        # weights = np.ones(shape = weights.shape)

        train_set_weights = theano.shared(np.asarray(weights, dtype = np.float32), borrow = True)


        valid_set_indices =np.where(session == unique_sessions[ix_out])[1].astype(int)
        x_valid = x_data[valid_set_indices,:]

        session_valid = session[:,valid_set_indices]
        y_valid = y_data[0,valid_set_indices]


        train_set_x, train_set_y = x_train, y_train
        valid_set_x, valid_set_y = x_valid, y_valid

        train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))
        valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y))


        batch_size = 12

        # build model
        x = T.dmatrix("x")
        y = T.ivector("y")
        weights = T.fvector('weights')

        index = T.lscalar('ind')
        n_train_batches = train_set_x.get_value(borrow = True).shape[0]//batch_size
        n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]//batch_size
        #n_test_batches = test_set_x.get_value(borrow = True).shape[0]//batch_size
        #L2_reg = 1.0/7.2*10**4/len(y_train)
        #L2_reg  = 10



        print 'n train batches', n_train_batches
        n_feature = x_train.shape[1]
        n_class = 2

        C = 7.2*1e-4/n_train_batches*12
        L2_reg = 1.0/C
        C = 1

        classifier = LogisticRegression(input=x, n_feature = n_feature, n_class = n_class, weights= weights)
        cost =C*classifier.negative_log_likelihood(y) + classifier.L2_cost*L2_reg*0.5
        gw, gb = T.grad(cost, [classifier.W,classifier.b])
        learning_rate = 0.01



        updates = [(classifier.W, classifier.W-learning_rate*gw),(classifier.b, classifier.b-learning_rate*gb)]
        train_model = theano.function(inputs = [index], outputs = classifier.errors(y), updates = updates,
                                      givens=[(x, train_set_x[index*batch_size : (index+1)*batch_size])
                                          ,(y, train_set_y[index*batch_size : (index+1)*batch_size]),
                                              (weights, train_set_weights[index * batch_size: (index + 1) * batch_size])])




        validate_model =theano.function(inputs = [index], outputs = [classifier.errors(y), classifier.p_1],
                                        givens = [(x, valid_set_x[index*batch_size : (index+1)*batch_size])
                                          ,(y, valid_set_y[index*batch_size : (index+1)*batch_size])])

        # test_model =theano.function(inputs = [index], outputs = classifier.errors(y),
        #                                 givens = [(x, test_set_x[index*batch_size : (index+1)*batch_size])
        #                                   ,(y, test_set_y[index*batch_size : (index+1)*batch_size])])
        #


        # traing models

        print '... training the model'
        patience = 10000
        patience_increase = 10
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
                    #this_validation_loss = np.mean(validation_losses)
                    this_validation_loss = np.mean([loss[0] for loss in validation_losses])
                    y_valid = valid_set_y.get_value()
                    prob = list(itertools.chain.from_iterable([loss[1][:,1] for loss in validation_losses]))
                    fpr, tpr, thresholds = metrics.roc_curve(y_valid, prob, pos_label = 1)
                    auc = metrics.auc(fpr,tpr)

                    #test_losses = [test_model(i) for i in range(n_test_batches)]
                    #test_score = np.mean(test_losses)
                    #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'%(epoch, minibatch_index+1,n_train_batches, test_score*100)

                    #print 'epoch %i, minibatch %i/%i, validation error %f %%'%(epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
                    #print 'epoch %i, minibatch %i/%i, validation AUC %f %%'%(epoch, minibatch_index+1, n_train_batches, auc*100)


                    #test_losses = [test_model(i) for i in range(n_test_batches)]
                    #test_score = np.mean(test_losses)
                    #print 'epoch %i, minibatch %i/%i, test error of best model %f %%'%(epoch, minibatch_index+1,n_train_batches, test_score*100)

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
        prob_session.append(prob)
        y_session.append(y_valid)


        end_time = timeit.default_timer()

        print 'The code run for %d epochs, with %f epochs/sec'%(epoch, 1.*epoch/(end_time-start_time))

        auc_session[ii] = auc


    prob_session = list(itertools.chain.from_iterable([prob for prob in prob_session]))
    y_session = list(itertools.chain.from_iterable([z for z in y_session]))


    fpr, tpr, thresholds = metrics.roc_curve(y_session, prob_session, pos_label = 1)
    auc = metrics.auc(fpr,tpr)
    lambda_grid[k] = L2_reg
    auc_grid[k] = auc

    print 'auc: {0}, L2_reg: {1}'.format(auc, L2_reg)

    #print auc_session
    #print np.mean(auc_session)


from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.log(lambda_grid), auc_grid, color = 'red')
