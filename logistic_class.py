# Logistic Regression Class
# mini-batch gradient descent MBGD

import theano
import theano.tensor as T
import numpy as np
import timeit

class LogisticRegression(object):
    def __init__(self, input, n_feature, n_class):
        self.W = theano.shared(value = np.zeros((n_feature,n_class), dtype = theano.config.floatX), name = 'W', borrow = True)
        self.b = theano.shared(value = np.zeros((1,n_class), dtype = theano.config.floatX), name = 'b', borrow = True, broadcastable = (True,False))
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






def shared_dataset(data_xy, borrow = True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype = np.int32), borrow = borrow)
    return shared_x, shared_y





# get dataset

train_set_x, train_set_y = shared_dataset((X[:10000,:],y_sim[:10000]))
valid_set_x, valid_set_y = shared_dataset((X[10000:15000,:],y_sim[10000:15000]))
test_set_x, test_set_y = shared_dataset((X[15000:,:],y_sim[15000:]))


batch_size = 10


# build model
x = T.dmatrix("x")
y = T.ivector("y")
index = T.lscalar('ind')
n_train_batches = train_set_x.get_value(borrow = True).shape[0]//batch_size
n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]//batch_size
n_test_batches = test_set_x.get_value(borrow = True).shape[0]//batch_size


classifier = LogisticRegression(input = x, n_feature = 10, n_class = 10)
cost = classifier.negative_log_likelihood(y)
gw, gb = T.grad(cost, [classifier.W,classifier.b])
learning_rate = 0.3

updates = [(classifier.W, classifier.W-learning_rate*gw),(classifier.b, classifier.b-learning_rate*gb)]
train_model = theano.function(inputs = [index], outputs = classifier.errors(y), updates = updates,
                              givens=[(x, train_set_x[index*batch_size : (index+1)*batch_size])
                                  ,(y, train_set_y[index*batch_size : (index+1)*batch_size])])




validate_model =theano.function(inputs = [index], outputs = classifier.errors(y),
                                givens = [(x, valid_set_x[index*batch_size : (index+1)*batch_size])
                                  ,(y, valid_set_y[index*batch_size : (index+1)*batch_size])])

test_model =theano.function(inputs = [index], outputs = classifier.errors(y),
                                givens = [(x, test_set_x[index*batch_size : (index+1)*batch_size])
                                  ,(y, test_set_y[index*batch_size : (index+1)*batch_size])])



# traing model

print '... training the model'
patience = 5000
patience_increase = 10
improvement_threshold = 0.99999999999999999999999999999995
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



