# Recurrent neural net for EEG data
# Each list consists of 12 items
# Lists are assumed to be independent
# The model
# a(t) = b + W h(t-1) + U x(t)
# h(t) = tanh(a(t))
# o(t) = c + V h(t)
# y_hat(t) = softmax(o(t))
# L(t) = -log P(y(t) | y_hat(t))
# Lasagne Implementation

import lasagne

import theano
import theano.tensor as T
import numpy as np
import urllib
import gzip
import os
import pickle
import timeit
import scipy.io as sio
import itertools
import pandas as pd
from sklearn import metrics
import sys
import collections
from sklearn.externals import joblib


from rnn_class import*

N_HIDDEN = 500

def RNN(input_var = None):
    n_features = input_var.shape[1]
    n_samp = input_var.shape[0]
    n_batch = 10
    l_in = lasagne.layers.InputLayer(shape = (n_batch, n_samp, n_features), input_var = input_var)
    l_in_to_hid = lasagne.layers.RecurrentLayer(l_in, num_units = N_HIDDEN , nonlinearity = lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_in_to_hid, num_units = 2, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out



rnn_network = RNN(x_train)



prediction = lasagne.layers.get_ouptut(rnn_network)
loss = lasagne.obje