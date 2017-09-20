# lasagne tutorial
import numpy as np
import theano
import theano.tensor as T
import lasagne


def build_mlp(input_var = None):
    l_in = lasagne.layers.InputLayer(shape = (None, 1, 28, 28), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p = 0.2)
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units = 800, nonlinearity = lasagne.nonlinearities.rectify, W = lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p = 0.5)
    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units = 800, p = 0.5)
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p = 0.5)
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units = 10, nonlinearity = lasagne.nonlinearities.softmax)
    return l_out


# build_cnn()

def build_cnn(input_var = None):
    network = lasagne.layers.InputLayer(shape = (None,1,28,28), input_var = input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 32, fitler_size = (5,5), nonliearity = lasagne.nonlinearities.rectifity,
                                         W = lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = (2,2))
    network = lasagne.layers.Conv2DLayer(network, num_filters = 32, filter_size = (5,5),
                                         nonlinearity = lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = (2,2))

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p = .5), num_units = 256,
                                        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)



input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = build_mlp(input_var)


