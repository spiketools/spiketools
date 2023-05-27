"""Tests for spiketools.utils.timestamps"""

import numpy as np

from spiketools.utils.timestamps import *

###################################################################################################
###################################################################################################

def test_compute_sample_durations():

    # define a timestamp, with irregular times
    timestamp = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0])
    bin_time = compute_sample_durations(timestamp)

    # check dimensions & sum
    assert bin_time.shape[0] == timestamp.shape[0]
    assert np.sum(np.diff(timestamp)) == np.sum(bin_time)

def test_infer_time_unit(tspikes):

    # Check test data in seconds
    inferred = infer_time_unit(tspikes)
    assert inferred == 'seconds'

    # Check test data in milliseconds
    inferred = infer_time_unit(tspikes * 1000)
    assert inferred == 'milliseconds'

def test_convert_ms_to_sec():

    value = 1250
    output1 = convert_ms_to_sec(value)
    assert output1 == 1.250

    array = np.array([1000, 1500, 2000])
    output2 = convert_ms_to_sec(array)
    assert np.array_equal(output2, np.array([1.0, 1.5, 2.0]))

def test_convert_sec_to_min():

    value = 30
    output1 = convert_sec_to_min(value)
    assert output1 == 0.5

    array = np.array([30, 60, 120])
    output2 = convert_sec_to_min(array)
    assert np.array_equal(output2, np.array([0.5, 1.0, 2.0]))

def test_convert_min_to_hour():

    value = 30
    output1 = convert_min_to_hour(value)
    assert output1 == 0.5

    array = np.array([30, 60, 90])
    output2 = convert_min_to_hour(array)
    assert np.array_equal(output2, np.array([0.5, 1.0, 1.5]))

def test_convert_ms_to_min():

    value = 30000
    output1 = convert_ms_to_min(value)
    assert output1 == 0.5

    array = np.array([30000, 60000, 120000])
    output2 = convert_ms_to_min(array)
    assert np.array_equal(output2, np.array([0.5, 1.0, 2.0]))

def test_convert_nsamples_to_time():

    n_samples = 12
    fs = 500

    out = convert_nsamples_to_time(n_samples, fs)
    assert isinstance(out, float)
    assert out == 0.024

def test_convert_time_to_nsamples():

    time = 0.024
    fs = 500

    out = convert_time_to_nsamples(time, fs)
    assert isinstance(out, int)
    assert out == 12

def test_sum_time_ranges():

    range0 = []
    out0 = sum_time_ranges(range0)
    assert out0 == 0

    range1 = [0, 15.5]
    out1 = sum_time_ranges(range1)
    assert out1 == 15.5

    ranges = [[0, 5.5], [10, 14.5]]
    out2 = sum_time_ranges(ranges)
    assert out2 == 10.0

def test_create_bin_times():

    # Check basic time bin definition
    bins1 = np.array([0, 0.5, 1.0, 1.5, 2.0])
    out1 = create_bin_times(bins1)
    assert np.array_equal(out1, np.array([0.25, 0.75, 1.25, 1.75]))

    # Check bin definition that doesn't start at zero
    bins2 = np.array([-1.0, -0.5, 0, 0.5, 1.0])
    out2 = create_bin_times(bins2)
    assert np.array_equal(out2, np.array([-0.75, -0.25, 0.25, 0.75]))

    # Check uneven bin definition
    bins2 = np.array([0, 0.5, 1.0, 2.0])
    out3 = create_bin_times(bins2)
    assert np.array_equal(out3, np.array([0.25, 0.75, 1.5]))

def test_split_time_value():

    value = 3600 + 1800 + 30
    hours, minutes, seconds = split_time_value(value)
    assert (hours, minutes, seconds) == (1, 30, 30)

def test_format_time_string():

    hours, minutes, seconds = 1.0, 30.0, 45.0
    output = format_time_string(hours, minutes, seconds)
    for el in (hours, minutes, seconds):
        assert str(el) in output
