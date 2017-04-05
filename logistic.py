import theano
import numpy as np
import theano.tensor as T
rng = np.random


# simulate data
N = 400
feats = 784
D = (rng.randn(N,feats), rng.randint(size = N, low=0, high=2))

x = T.dmatrix("x")
y = T.dvector("y")
w = theano.shared(rng.randn(feats), name = 'w')
b = theano.shared(0.0, name='b')
p_1 = 1/(1 + T.exp(-T.dot(x,w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1+0.0000001) - (1-y) * T.log(1-p_1-0.0000000001)
cost = xent.mean() + 0.01*(w**2).sum()
gw, gb = T.grad(cost, [w,b])

train = theano.function(inputs = [x,y], outputs = [prediction, xent.mean()], updates = ((w, w-0.01*gw), (b, b-0.01*gb)))
predict = theano.function(inputs =[x], outputs = prediction)


training_steps = 10000
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print 'Training error: {0:3}'.format(err)



x = T.dvector("x")
y = T.dscalar("y")
w = theano.shared(rng.randn(feats), name = 'w')
b = theano.shared(0.0, name='b')
p_1 = 1/(1 + T.exp(-T.dot(x,w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1+0.0000001) - (1-y) * T.log(1-p_1-0.0000000001)
cost = xent.mean() + 0.01*(w**2).sum()
gw, gb = T.grad(cost, [w,b])

train_sgd = theano.function(inputs = [x,y], outputs = [prediction, xent.mean()], updates = ((w, w-0.1*gw/feats), (b, b-0.1*gb/feats)))
predict = theano.function(inputs =[x], outputs = prediction)


training_steps = 1000
for i in range(training_steps):
    err = 0
    for j in range(N):
        pred, err_sgd = train_sgd(D[0][j,:], D[1][j])
        err = err + err_sgd
    print 'Training error: {0:3}'.format(err/N)
