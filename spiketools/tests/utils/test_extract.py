"""Tests for spiketools.utils.extract"""

import numpy as np

from spiketools.utils.extract import *

###################################################################################################
###################################################################################################

def test_get_range():

    data = np.array([0.5, 1., 1.5, 2., 2.5])

    out1 = get_range(data, min_value=1.)
    assert np.array_equal(out1, np.array([1., 1.5, 2., 2.5]))

    out2 = get_range(data, max_value=2.)
    assert np.array_equal(out2, np.array([0.5, 1., 1.5, 2.]))

    out3 = get_range(data, min_value=1., max_value=2.)
    assert np.array_equal(out3, np.array([1., 1.5, 2.]))

    out4 = get_range(data, min_value=1., max_value=2., reset=1.)
    assert np.array_equal(out4, np.array([0., 0.5, 1.0]))

def test_get_value_by_time():

    times = np.array([1, 2, 3, 4, 5])
    values_1d = np.array([5, 8, 4, 6, 7])
    values_2d = np.array([[5, 8, 4, 6, 7], [5, 8, 4, 6, 7]])

    value_out = get_value_by_time(times, values_1d, 3)
    assert value_out == values_1d[2]

    value_out = get_value_by_time(times, values_1d, 3.4)
    assert value_out == values_1d[2]

    value_out = get_value_by_time(times, values_2d, 3)
    assert np.array_equal(value_out, values_2d[:, 2])

def test_get_values_by_times():

    times = np.array([1, 2, 3, 4, 5])
    values_1d = np.array([5, 8, 4, 6, 7])
    values_2d = np.array([[5, 8, 4, 6, 7], [5, 8, 4, 6, 7]])

    timepoints = np.array([1.75, 4.15])

    outputs = get_values_by_times(times, values_1d, timepoints)
    assert len(outputs) == len(timepoints)
    assert np.array_equal(outputs, np.array([8, 6]))

    outputs = get_values_by_times(times, values_2d, timepoints)
    assert len(outputs) == len(timepoints)
    assert np.array_equal(outputs, np.array([[8, 6], [8, 6]]))

def test_get_values_by_time_range():

    times = np.array([1, 2, 3, 4, 5])
    values = np.array([5, 8, 4, 6, 7])

    times_out, values_out = get_values_by_time_range(times, values, 2, 4)
    assert np.array_equal(times_out, np.array([2, 3, 4]))
    assert np.array_equal(values_out, np.array([8, 4, 6]))
