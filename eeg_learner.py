# CCN for iEEG signals
import numpy as np
import theano
import theano.tensor as T
import lasagne


def build_eeg_learner(input_var = None, n_chans = 1, n_freq = 8, n_time = 0):
    l_in = lasagne.layers.InputLayer(shape = (None, n_chans, n_freq, n_time))



