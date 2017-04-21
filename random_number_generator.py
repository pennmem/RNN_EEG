import theano
import numpy as np
import theano.tensor as T
from theano import function

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

srng = RandomStreams(234)


rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
rv_x = srng.uniform((2,2))



f = function([], rv_u)
h = function([], rv_x)
g = function([], rv_n, no_default_updates = True)
nearly_zeros = function([], rv_u + rv_u - 2*rv_u)



state_value = rv_u.rng.get_value()
rv_x.rng.set_value(state_value)

print h()
print f()

state_before_u0 = rv_u.rng.get_value().get_state()
print nearly_zeros()
print f()
rv_u.rng.get_value(borrow = True).set_state(state_before_u0)
