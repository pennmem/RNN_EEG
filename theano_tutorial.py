import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano import pp
from theano import scan
import time, numpy as np

srng = RandomStreams(seed = 234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates = True)
nearly_zeros = function([], rv_u  + rv_u - 2*rv_u)

# seeding seed
rng_val = rv_u.rng.get_value(borrow = True)
rng_val.seed(89234)
rv_u.rng.set_value(rng_val)


srng.seed(100)

state_after_v0 = rv_u.rng.get_value().get_state()
f()
f()

nearly_zeros()
v1 = f()
rng = rv_u.rng.get_value(borrow = True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow = True)

v2 = f()
v3 = f()


# Derivatives

x = T.dscalar('x')
y = x**2
gy = T.grad(y,x)
pp(gy)

f = function([x], gy)
f(4)

# Jacobian
x = T.dvector('x')
y = x ** 2
J, updates = scan(lambda i, y, x: T.grad(y[i],x), sequences = T.arange(y.shape[0]), non_sequences = [y,x])
f = function([x], J)

f([4,4])

# ifelse and switch
from theano.ifelse import ifelse
a,b = T.scalars('a', 'b')
x,y = T.matrices('x', 'y')
z_switch = T.switch(T.lt(a,b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a,b), T.mean(x), T.mean(y))
f_switch = function([a,b,x,y], z_switch)
f_lazyifelse = function([a,b,x,y], z_lazy)


n_times = 10

val1 = 0.
val2 = 1.
big_mat1 = np.ones((10000, 1000))
big_mat2 = np.ones((10000, 1000))

tic = time.clock()
for i in range(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print('time spent evaluating both values %f sec' % (time.clock() - tic))



tic = time.clock()
for i in range(n_times):
    f_lazyifelse(val1, val2, big_mat1, big_mat2)
print('time spent evaluating one value %f sec' % (time.clock() - tic))

# loop
X = T.matrix('X')
W = T.matrix('W')
b_sym = T.vector('b_sym')
results, updates = scan(lambda v: T.tanh(T.dot(v,W) + b_sym), sequences = X)
compute_elementwise = function(inputs = [X,W,b_sym], outputs= results)


x = np.eye(2, dtype=T.config.floatX)
w = np.ones((2, 2), dtype=T.config.floatX)
b = np.ones((2), dtype=T.config.floatX)
b[1] = 2




# define tensor variables
X = T.vector("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

results, updates = scan(lambda y, p, x_tm1: T.tanh(T.dot(x_tm1, W) + T.dot(y, U) + T.dot(p, V)),
          sequences=[Y, P[::-1]], outputs_info=[X])
compute_seq = function(inputs=[X, W, Y, U, P, V], outputs=results)

# test values
x = np.zeros((2), dtype=T.config.floatX)
x[1] = 1
w = np.ones((2, 2), dtype=T.config.floatX)
y = np.ones((5, 2), dtype=T.config.floatX)
y[0, :] = -3
u = np.ones((2, 2), dtype=T.config.floatX)
p = np.ones((5, 2), dtype=T.config.floatX)
p[0, :] = 3
v = np.ones((2, 2), dtype=T.config.floatX)

print(compute_seq(x, w, y, u, p, v))


k = T.iscalar('k')
A = T.vector('A')
result, updates = scan(fn = lambda prior_result, A: prior_result*A, outputs_info= [dict(initial = T.ones_like(A)*2, taps = [-1])], non_sequences = A, n_steps = k)

final_result = result[-1]

power = function(inputs = [A,k], outputs = final_result, updates = updates)
power(np.ones(2,)*2, 10)


# Example: calculating a polynomial
coefficients = T.vector("coefs")
x = T.scalar('x')
max_coef_supported = 10000


def compute_polynomial(coefs, power, free_variable):
    return coefs*(free_variable**power)


components, updates = scan(fn = compute_polynomial, outputs_info= None, sequences = [coefficients, T.arange(max_coef_supported)], non_sequences= x)

polynomial = components.sum()

calculate_polynomial = function(inputs =[coefficients, x], outputs= polynomial,updates = updates)


test_coefficients = np.asarray([1, 0, 2], dtype=np.float32)
test_value = 3
print(calculate_polynomial(test_coefficients, test_value))
print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))












































