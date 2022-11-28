"""Tests for spiketools.spatial.utils"""

import numpy as np

from spiketools.spatial.utils import *

###################################################################################################
###################################################################################################

def test_compute_nbins():

    # check 1d case
    out1 = compute_nbins([5, 5])
    assert out1 == 25

    # check 2d case
    out2 = compute_nbins([5])
    out3 = compute_nbins(5)
    assert out2 == out3 == 5

def test_compute_pos_ranges():

    positions = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])

    ranges = compute_pos_ranges(positions)
    ranges[0] == [1, 5]
    ranges[1] == [5, 9]

def test_compute_sample_durations():

    # define a timestamp, with irregular times
    timestamp = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0])
    bin_time = compute_sample_durations(timestamp)

    # check dimensions & sum
    assert bin_time.shape[0] == timestamp.shape[0]
    assert np.sum(np.diff(timestamp)) == np.sum(bin_time)

def test_compute_bin_width():

    bins = [1., 2., 3., 4., 5.]
    binw = compute_bin_width(bins)
    assert binw == 1.

def test_convert_2dindices():

    bins = [3, 2]
    xbins = np.array([1, 0, 2, 1])
    ybins = np.array([1, 0, 1, 0])
    expected = np.array([4, 0, 5, 1])

    inds = convert_2dindices(xbins, ybins, bins)
    assert isinstance(inds, np.ndarray)
    assert np.array_equal(inds, expected)

def test_convert_1dindices():

    bins = [3, 2]
    indices = np.array([4, 0, 5, 1])
    expected_x, expected_y = np.array([1, 0, 2, 1]), np.array([1, 0, 1, 0])

    xbins, ybins = convert_1dindices(indices, bins)
    assert isinstance(xbins, np.ndarray)
    assert isinstance(ybins, np.ndarray)
    assert np.array_equal(xbins, expected_x)
    assert np.array_equal(ybins, expected_y)
