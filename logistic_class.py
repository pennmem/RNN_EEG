# Logistic Regression Class
# mini-batch gradient descent MBGD

import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    def __init__(self, input, n_input, n_feature, n_class):
        self.W = theano.shared(value = np.zeros((n_input,n_feature), dtype = theano.config.floatX), name = 'W', borrow = True)
        self.b = theano.shared(value = np.zeros((1,n_class), dtype = theano.config.floatX), name = 'b', borrow = True)
        self.input = input
        self.p_1 = T.nnet.softmax(T.dot(self.input,self.W) + self.b)
        self.y_pred = T.argmax(self.p_1, axis = 1) # find argmax
        self.params = [self.W, self.b]


    def negative_log_likelihood(self,y):
        # cost function
        return -T.sum(y*T.log(self.p_1),1)


    def errors(self,y):
        return T.mean(T.neq(self.y_pred,y))




x = T.dmatrix("x")
y = T.dmatrix("y")
classifier = LogisticRegression(input = x, n_input = 1000, n_feature = 10, n_class = 3)
cost = classifier.negative_log_likelihood(y).mean()
gw, gb = T.grad(cost, [classifier.W,classifier.b])
learning_rate = 0.1
updates = [(classifier.W, classifier.W-learning_rate*gw),(classifier.b, classifier.b-learning_rate*gb)]
batch_size = 10
index = 10


train_model = theano.function(inputs = [index], outputs = cost, updates = updates,
                              givens=[(x, train_set_x[index*batch_size : (index+1)*batch_size])
                                  ,((y, train_set_y[index*batch_size : (index+1)*batch_size]))])







