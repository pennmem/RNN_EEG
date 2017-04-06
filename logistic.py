import theano
import numpy as np
import theano.tensor as T
rng = np.random

def softmax(x):
    x = x - np.max(x)
    output = np.exp(x)/(np.exp(x).sum())
    return output

def sample_y(p):
    return np.random.multinomial(1,p,size = 1)

def find_class(y):
    return np.where(y > 0)

# simulate data
N = 1000
feats = 10
n_class = 3
X = rng.randn(N,feats)
W = rng.randn(feats,n_class)*0.2
b = np.array([-0.5,0,0.3])
o = np.apply_along_axis(softmax,1,np.dot(X,W) + b)
y = np.apply_along_axis(sample_y,1,o)
y = np.argmax(y, axis = 1)

D = (X,y)

# build model
x = T.dmatrix("x")
y = T.dmatrix("y")
w = theano.shared(rng.randn(feats,n_class), name = 'w')
b = theano.shared(value = np.zeros((1,n_class)), name='b', broadcastable = (True,False))
p_1 = T.nnet.softmax(T.dot(x,w)+b)
prediction = T.argmax(p_1)
#xent = -y * T.log(p_1+0.0000001) - (1-y) * T.log(1-p_1-0.0000000001)
xent = -T.sum(y*T.log(p_1),1)
cost = xent.mean()
gw, gb = T.grad(cost, [w,b])

train = theano.function(inputs = [x,y], outputs = [prediction, xent.mean()], updates = ((w, w-0.1*gw), (b, b-0.1*gb)))
predict = theano.function(inputs =[x], outputs = prediction)
training_steps = 10000
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    if i%100 == 0:
        print 'Training cost after {0:3} iterations is: {1:3}'.format(i,err)




import numpy as np

a = np.random.random((2,2))
b = np.random.random((2,2))
c = np.tensordot(a,b,[[1],[1]])



# using shared variable
state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates = [(state, state+inc)])

fn_of_state = state*2 + inc
foo = T.scalar(dtype =  state.dtype)
skip_shared = theano.function([inc, foo], fn_of_state, givens = [(state, foo)])
skip_shared(1,30)
print state.get_value()


