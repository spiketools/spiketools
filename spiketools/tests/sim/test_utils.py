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

def test_apply_refractory_train():

    train1 = np.array([0, 1, 1, 1, 0, 1, 0])
    refractory_samples1 = 1
    train_out1 = apply_refractory_train(train1, refractory_samples1)
    assert train_out1.shape == train1.shape
    assert np.isin(train1, [0, 1]).all()
    assert np.array_equal(train_out1, np.array([0, 1, 0, 1, 0, 1, 0]))

    train2 = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    refractory_samples2 = 2
    train_out2 = apply_refractory_train(train2, refractory_samples2)
    assert train_out2.shape == train2.shape
    assert np.isin(train2, [0, 1]).all()
    assert np.array_equal(train_out2, np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]))

def test_refractory_times():
    # Tests the 'refractory' decorator, applied to 'times' functions

    @apply_refractory('times')
    def _spike_times():
        return np.array([0.100, 0.5100, 0.5105, 0.75, 0.95])

    # test without passing in refractory time
    out = _spike_times(refractory=None)
    assert np.array_equal(out, np.array([0.100, 0.5100, 0.5105, 0.75, 0.95]))

    # test with passing in refractory time
    out = _spike_times(refractory=0.001)
    assert np.array_equal(out, np.array([0.100, 0.5100, 0.75, 0.95]))

    # test with accessing refractory time from function default
    @apply_refractory('times')
    def _spike_times2(refractory=0.001):
        return np.array([0.100, 0.5100, 0.5105, 0.75, 0.95])
    out = _spike_times2()
    assert np.array_equal(out, np.array([0.100, 0.5100, 0.75, 0.95]))

def test_refractory_train():
    # Tests the 'refractory' decorator, applied to 'train' functions

    @apply_refractory('train')
    def _spike_train():
        return np.array([0, 1, 1, 0, 1])

    # test without passing in refractory samples
    out = _spike_train(refractory=None)
    assert np.array_equal(out, np.array([0, 1, 1, 0, 1]))

    # test with passing in refractory samples
    out = _spike_train(refractory=1)
    assert np.array_equal(out, np.array([0, 1, 0, 0, 1]))

    # test with accessing refractory time from function default
    @apply_refractory('train')
    def _spike_train2(refractory=1):
        return np.array([0, 1, 1, 0, 1])
    out = _spike_train2()
    assert np.array_equal(out, np.array([0, 1, 0, 0, 1]))
