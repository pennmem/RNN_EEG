# Train an autoencoder of the data
# Another dimensionality technique
import matplotlib as plt
import theano
import numpy as np
import lasagne


NUM_FEATURES = 8*N_ELEC
NUM_UNITS = 50 # hard code for now

l_in = lasagne.layers.InputLayer(shape = (None, NUM_FEATURES))
encoder_l_out = lasagne.layers.DenseLayer(l_in, num_units)






