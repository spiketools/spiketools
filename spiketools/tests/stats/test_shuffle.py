"""Tests for spiketools.stats.shuffle"""

import numpy as np

from spiketools.utils import set_random_seed

from spiketools.stats.shuffle import *

###################################################################################################
###################################################################################################

def test_shuffle_spikes(tspikes):

    approaches = ['ISI', 'BINCIRC', 'POISSON', 'CIRCULAR']
    kwargs = [{}, {}, {}, {'shuffle_min' : 10}]

    for approach, kwarg in zip(approaches, kwargs):
        shuffled = shuffle_spikes(tspikes, approach=approach, n_shuffles=1, **kwarg)
        assert isinstance(shuffled, np.ndarray)

def test_shuffle_isis(tspikes):

    shuffled = shuffle_isis(tspikes, n_shuffles=1)
    assert isinstance(shuffled, np.ndarray)
    assert tspikes.shape[-1] == shuffled.shape[-1]
    assert not np.array_equal(tspikes, shuffled)

    # Test that get a different answer with different random states
    set_random_seed(12)
    out1 = shuffle_isis(tspikes, n_shuffles=1)
    set_random_seed(21)
    out2 = shuffle_isis(tspikes, n_shuffles=1)
    assert not np.array_equal(out1, out2)

    # Test with more shuffles
    n_shuffles = 5
    out_many = shuffle_isis(tspikes, n_shuffles=n_shuffles)
    assert out_many.shape == (n_shuffles, len(tspikes))

def test_shuffle_bins(tspikes):

    shuffled = shuffle_bins(tspikes, n_shuffles=1)
    assert isinstance(shuffled, np.ndarray)
    assert tspikes.shape[-1] == shuffled.shape[-1]
    assert not np.array_equal(tspikes, shuffled)

    # Test that get a different answer with different random states
    set_random_seed(12)
    out1 = shuffle_bins(tspikes, n_shuffles=1)
    set_random_seed(21)
    out2 = shuffle_bins(tspikes, n_shuffles=1)
    assert not np.array_equal(out1, out2)

    # Test with more shuffles
    n_shuffles = 5
    out_many = shuffle_bins(tspikes, n_shuffles=n_shuffles)
    assert out_many.shape == (n_shuffles, len(tspikes))


def test_shuffle_poisson(tspikes):
    # TO FIX / FINISH

    shuffled = shuffle_poisson(tspikes)
    assert isinstance(shuffled, np.ndarray)
    #assert tspikes.shape[-1] == shuffled.shape[-1]
    #assert not np.array_equal(tspikes, shuffled)

    # Test that get a different answer with different random states
    #out1 = shuffle_poisson(tspikes, n_shuffles=1, random_state=12)
    #out2 = shuffle_poisson(tspikes, n_shuffles=1, random_state=21)
    #assert not np.array_equal(out1, out2)

    # Test with more shuffles
    #n_shuffles = 5
    #out_many = shuffle_poisson(tspikes, n_shuffles=n_shuffles)
    #assert out_many.shape == (n_shuffles, len(tspikes))

def test_shuffle_circular(tspikes):

    shuffled = shuffle_circular(tspikes, 10, n_shuffles=1)
    assert isinstance(shuffled, np.ndarray)
    assert tspikes.shape[-1] == shuffled.shape[-1]
    assert not np.array_equal(tspikes, shuffled)

    # Test that get a different answer with different random states
    set_random_seed(12)
    out1 = shuffle_circular(tspikes, 10, n_shuffles=1)
    set_random_seed(21)
    out2 = shuffle_circular(tspikes, 10, n_shuffles=1)
    assert not np.array_equal(out1, out2)

    # Test with more shuffles
    n_shuffles = 5
    out_many = shuffle_circular(tspikes, 10, n_shuffles=n_shuffles)
    assert out_many.shape == (n_shuffles, len(tspikes))
