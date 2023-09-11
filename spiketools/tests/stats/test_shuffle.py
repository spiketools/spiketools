"""Tests for spiketools.stats.shuffle"""

import numpy as np

from spiketools.utils import set_random_seed
from spiketools.utils.extract import drop_range

from spiketools.stats.shuffle import *

###################################################################################################
###################################################################################################

def test_shuffle_spikes(tspikes):

    approaches = ['ISI', 'CIRCULAR', 'BINCIRC']
    kwargs = [{}, {'shuffle_min' : 10}, {}]

    for approach, kwarg in zip(approaches, kwargs):
        shuffled = shuffle_spikes(tspikes, approach=approach, n_shuffles=1, **kwarg)
        assert isinstance(shuffled, np.ndarray)

def test_drop_shuffle_range():

    # Mock shuffle function
    @drop_shuffle_range
    def _shuffle_spikes(spikes, n_shuffles=2):
        return np.array([spikes + 1] * n_shuffles)

    n_shuffles = 2
    spikes = np.array([0.5, 1.5, 1.9, 4.1, 5.4, 5.9])

    # Check without activating decorator
    out = _shuffle_spikes(spikes, n_shuffles)
    assert np.allclose(out, np.array([spikes + 1] * n_shuffles))

    # Check with applying a drop range
    #   This test the spikes that get moved from the empty range (spikes @ ind: [1, 2])
    drop_time_range = [2, 4]
    out = _shuffle_spikes(spikes, n_shuffles, drop_time_range=drop_time_range)
    toutput = np.array([0.5, 3.5, 3.9, 4.1, 5.4, 5.9])
    assert np.allclose(out, np.array([toutput + 1] * n_shuffles))

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

    shuffled = shuffle_poisson(tspikes)
    assert isinstance(shuffled, list)
    assert not np.array_equal(tspikes, shuffled)

    # Test that get a different answer with different random states
    out1 = shuffle_poisson(tspikes, n_shuffles=1)
    out2 = shuffle_poisson(tspikes, n_shuffles=1)
    assert not np.array_equal(out1, out2)

    # Test with more shuffles
    #   Note: checks the # of shuffles, but not the exact # of spikes, which is not guaranteed
    n_shuffles = 5
    out_many = shuffle_poisson(tspikes, n_shuffles=n_shuffles)
    assert len(out_many) == n_shuffles
