"""Tests for spiketools.utils.timestamps"""

import numpy as np

from spiketools.utils.timestamps import *

###################################################################################################
###################################################################################################

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

def test_split_time_value():

    value = 3600 + 1800 + 30
    hours, minutes, seconds = split_time_value(value)
    assert (hours, minutes, seconds) == (1, 30, 30)

def test_format_time_string():

    hours, minutes, seconds = 1.0, 30.0, 45.0
    output = format_time_string(hours, minutes, seconds)
    for el in (hours, minutes, seconds):
        assert str(el) in output
