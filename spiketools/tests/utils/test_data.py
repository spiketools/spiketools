"""Tests for spiketools.utils.data"""

import numpy as np

from spiketools.utils.data import *

###################################################################################################
###################################################################################################

def test_get_range():

    data = np.array([0.5, 1., 1.5, 2., 2.5])
    minv, maxv = get_range(data)
    assert isinstance(minv, float)
    assert isinstance(maxv, float)
    assert minv == 0.5
    assert maxv == 2.5

def test_restrict_range():

    data = np.array([0.5, 1., 1.5, 2., 2.5])

    out1 = restrict_range(data, min_time=1.)
    assert np.array_equal(out1, np.array([1., 1.5, 2., 2.5]))

    out2 = restrict_range(data, max_time=2.)
    assert np.array_equal(out2, np.array([0.5, 1., 1.5, 2.]))

    out3 = restrict_range(data, min_time=1., max_time=2.)
    assert np.array_equal(out3, np.array([1., 1.5, 2.]))

def test_get_value_by_time():

    times = np.array([1, 2, 3, 4, 5])
    values = np.array([5, 8, 4, 6, 7])

    value_out = get_value_by_time(times, values, 3)
    assert value_out == values[2]

    value_out = get_value_by_time(times, values, 3.4)
    assert value_out == values[2]

def test_get_value_by_time_range():

    times = np.array([1, 2, 3, 4, 5])
    values = np.array([5, 8, 4, 6, 7])

    times_out, values_out = get_value_by_time_range(times, values, 2, 4)
    assert np.array_equal(times_out, np.array([2, 3, 4]))
    assert np.array_equal(values_out, np.array([8, 4, 6]))
