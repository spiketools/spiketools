"""Tests for spiketools.utils.checks"""

import numpy as np

from pytest import raises, warns

from spiketools.utils.checks import *

###################################################################################################
###################################################################################################

def test_infer_time_unit(tspikes):

    # Check test data in seconds
    inferred = infer_time_unit(tspikes)
    assert inferred == 'seconds'

    # Check test data in milliseconds
    inferred = infer_time_unit(tspikes * 1000)
    assert inferred == 'milliseconds'

def test_check_time_bins(tspikes):

    # Check precomputed time bins
    tbins = np.arange(0, 10 + 0.5, 0.5)
    out = check_time_bins(tbins, tspikes)
    assert np.array_equal(tbins, out)

    # Check time bins given a time resolution, which should create same bins as precomputed
    out = check_time_bins(0.5, tspikes, trange=[0, 10])
    assert np.array_equal(tbins, out)

    # Check error & warning
    with raises(AssertionError):
        out = check_time_bins(np.array([1, 2, 1]), tspikes, trange=[0, 5])
    with warns(UserWarning):
        out = check_time_bins(0.5, tspikes, trange=[0, 5])

def test_check_position_bins():

    bins_int = 12
    bins_1d = [12]
    bins_2d = [5, 10]

    position = np.array([[1, 2, 3], [4, 5, 6]])

    out1d1 = check_position_bins(bins_int)
    out1d2 = check_position_bins(bins_1d)
    assert out1d1 == out1d2 == bins_1d

    bins = check_position_bins(bins_2d, position)
    assert bins == bins_2d

    with raises(AssertionError):
        check_position_bins([1.2, 2.3])
    with raises(AssertionError):
        check_position_bins([1, 2, 3])
    with raises(AssertionError):
        check_position_bins(bins_1d, position)
