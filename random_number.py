import random
import numpy as np
# #random.seed(10)
# np.random.seed(1000)
# print np.random.randint(10)
# print np.random.normal(0,1,1)

# np.random.seed(10)
# print np.random.randint(10)
# np.random.seed(100)
# print np.random.normal(0,1,1)
#
#
# np.random.seed(10)
# print np.random.randint(10)
# np.random.seed(100)
# print np.random.normal(0,1,1)
#
# np.random.seed(10)
# print np.random.randint(10)
# np.random.seed(100)
# print np.random.normal(0,1,1)
# #
# r = np.random.RandomState(10)
# rr = np.random.RandomState(100)
# print r.randint(10)
# print rr.normal(0,1,1)

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed = 234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))

# rng_val = rv_u.rng.get_value(borrow = True)
# rng_val.seed(89102)
# rv_u.rng.set_value(rng_val, borrow = True)
#


f = function([], rv_u)
g = function([], rv_n, no_default_updates= True)
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

# print f()
# print g()
# print nearly_zeros()


state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()
v1 = f()
print v1
rng = rv_u.rng.get_value(borrow = True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow  = True)
# print f()
# print f()

rng_val = rv_u.rng.get_value(borrow = True)
rng_values_before = rng_val.get_state()[1]

print f()

rng_val.seed(1083478)
rv_u.rng.set_value(rng_val)

print f()
rng_values_after = rng_val.get_state()[1]


print rng_values_after - rng_values_before
