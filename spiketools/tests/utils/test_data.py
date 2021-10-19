"""Tests for spiketools.utils.data"""

from spiketools.utils.data import *

###################################################################################################
###################################################################################################

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
