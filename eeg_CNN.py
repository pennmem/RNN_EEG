# Train an autoencoder of the data
# Another dimensionality technique
import matplotlib as plt
import theano
import numpy as np
import lasagne

def build_cnn(input_var, n_chans, n_freqs, n_times):
    network = lasagne.layers.InputLayer(shape = (None, n_chans, n_freqs, n_times))
    network = lasagne.layers.Conv1DLayer(network, num_filters=32, filter_size=(1,5), nonlinearity= lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(1,4))
    network = lasagne.layers.Conv1DLayer(network, num_filters = 32, filter_size = (1,5), nonlinearity= lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size = (1,4))
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network , p = 5), num_units=50,
                                        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p = .5))




