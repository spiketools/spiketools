"""Tests for spiketools.sim.utils"""

import numpy as np

from spiketools.sim.utils import *

###################################################################################################
###################################################################################################

def test_apply_refractory_times():

    times = np.array([0.512, 1.241, 1.242, 1.751, 2.124])
    refractory_time = 0.003

    times_out = apply_refractory_times(times, refractory_time)

    assert times_out.shape != times.shape
    assert np.array_equal(times_out, np.array([0.512, 1.241, 1.751, 2.124]))

def test_refractory_times():

    @refractory_times
    def _spike_times():
        return np.array([0.100, 0.5100, 0.5105, 0.75, 0.95])

    # test without passing in refractory time
    out = _spike_times(refractory=None)
    assert np.array_equal(out, np.array([0.100, 0.5100, 0.5105, 0.75, 0.95]))

    # test with passing in refractory time
    out = _spike_times(refractory=0.001)
    assert np.array_equal(out, np.array([0.100, 0.5100, 0.75, 0.95]))

    # test with accessing refractory time from function defaul
    @refractory_times
    def _spike_times2(refractory=0.001):
        return np.array([0.100, 0.5100, 0.5105, 0.75, 0.95])
    out = _spike_times2()
    assert np.array_equal(out, np.array([0.100, 0.5100, 0.75, 0.95]))

def test_apply_refractory_train():

    train = np.array([0, 1, 1, 0, 1])
    refractory_time = 0.002
    fs = 1000

    train_out = apply_refractory_train(train, refractory_time, fs)

    assert train_out.shape == train.shape
    assert np.isin(train, [0, 1]).all()
    assert np.array_equal(train_out, np.array([0, 1, 0, 0, 1]))
