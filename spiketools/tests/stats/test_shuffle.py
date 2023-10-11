"""Tests for spiketools.stats.shuffle"""

import numpy as np

from spiketools.utils import set_random_seed
from spiketools.utils.extract import drop_range

from spiketools.stats.shuffle import *

###################################################################################################
###################################################################################################

def test_shuffle_spikes(tspikes, tspikes_offset):

    n_shuffles = 2
    approaches = ['ISI', 'CIRCULAR', 'BINCIRC']
    kwargs = [{}, {'shuffle_min' : 10}, {}]

    for tdata in [tspikes, tspikes_offset]:
        for approach, kwarg in zip(approaches, kwargs):
            shuffled = shuffle_spikes(tdata, approach=approach, n_shuffles=n_shuffles, **kwarg)
            assert isinstance(shuffled, np.ndarray)
            assert len(shuffled) == n_shuffles
            for el in shuffled:
                assert len(el) == len(tdata)

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
    toutput = np.array([1.5, 4.5, 4.9, 5.1, 6.4, 6.9])
    assert np.allclose(out, np.array([toutput] * n_shuffles))

    # Test that this all still works with negative values / different start time
    spikes_off = np.array([-1.5, -1.1, -0.5, 0.1, 0.7, 4.1, 5.1])

    # Check without activating decorator
    out_off_no_drop = _shuffle_spikes(spikes_off, n_shuffles)
    assert np.allclose(out_off_no_drop, np.array([spikes_off + 1] * n_shuffles))

    # Check with activating decorator
    drop_time_range_off = [1, 4]
    out_off = _shuffle_spikes(spikes_off, n_shuffles, drop_time_range=drop_time_range_off)
    toutput_off = np.array([-0.5, -0.1, 0.5, 4.1, 4.7, 5.1, 6.1])
    assert np.allclose(out_off, np.array([toutput_off] * n_shuffles))

def test_shuffle_isis(tspikes, tspikes_offset):

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

    # Test spikes with start time
    shuffled_start = shuffle_isis(tspikes_offset, n_shuffles=1, start_time=-5)
    assert tspikes.shape[-1] == shuffled_start.shape[-1]
    assert np.min(shuffled_start) < 0

    # Test with more shuffles
    n_shuffles = 5
    out_many = shuffle_isis(tspikes, n_shuffles=n_shuffles)
    assert out_many.shape == (n_shuffles, len(tspikes))

def test_shuffle_circular(tspikes, tspikes_offset):

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

    # Test start time
    shuffled_start = shuffle_circular(tspikes_offset, 10, n_shuffles=1, start_time=-5)
    assert tspikes.shape[-1] == shuffled_start.shape[-1]
    assert np.min(shuffled_start) < 0

    # Test with more shuffles
    n_shuffles = 5
    out_many = shuffle_circular(tspikes, 10, n_shuffles=n_shuffles)
    assert out_many.shape == (n_shuffles, len(tspikes))

def test_shuffle_bins(tspikes, tspikes_offset):

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

    # Test start time
    shuffled_start = shuffle_bins(tspikes_offset, n_shuffles=1, start_time=-5)
    assert tspikes.shape[-1] == shuffled_start.shape[-1]
    assert np.min(shuffled_start) < 0

    # Test with more shuffles
    n_shuffles = 5
    out_many = shuffle_bins(tspikes, n_shuffles=n_shuffles)
    assert out_many.shape == (n_shuffles, len(tspikes))

def test_shuffle_poisson(tspikes, tspikes_offset):

    shuffled = shuffle_poisson(tspikes)
    assert isinstance(shuffled, list)
    assert not np.array_equal(tspikes, shuffled)

    # Test that get a different answer with different random states
    out1 = shuffle_poisson(tspikes, n_shuffles=1)
    out2 = shuffle_poisson(tspikes, n_shuffles=1)
    assert not np.array_equal(out1, out2)

    # Test start time
    shuffled_start = shuffle_poisson(tspikes_offset, n_shuffles=1, start_time=-5)
    assert np.min(shuffled_start[0]) < 0

    # Test with more shuffles
    #   Note: checks the # of shuffles, but not the exact # of spikes, which is not guaranteed
    n_shuffles = 5
    out_many = shuffle_poisson(tspikes, n_shuffles=n_shuffles)
    assert len(out_many) == n_shuffles
