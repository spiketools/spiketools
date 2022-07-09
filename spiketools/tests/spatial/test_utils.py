"""Tests for spiketools.spatial.utils"""

import numpy as np

from spiketools.spatial.utils import *

###################################################################################################
###################################################################################################

def test_compute_pos_ranges():

    positions = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])

    ranges = compute_pos_ranges(positions)
    ranges[0] == [1, 5]
    ranges[1] == [5, 9]

def test_compute_bin_width():

    bins = [1., 2., 3., 4., 5.]
    binw = compute_bin_width(bins)
    assert binw == 1.

def test_convert_2dindices():

    bins = [3, 2]
    xbins = np.array([1, 0, 2, 1])
    ybins = np.array([1, 0, 1, 0])
    expected = np.array([3, 0, 5, 2])

    inds = convert_2dindices(xbins, ybins, bins)
    assert isinstance(inds, np.ndarray)
    assert np.array_equal(inds, expected)

def test_convert_1dindices():

    bins = [3, 2]
    indices = np.array([3, 0, 5, 2])
    expected_x, expected_y = np.array([1, 0, 2, 1]), np.array([1, 0, 1, 0])

    xbins, ybins = convert_1dindices(indices, bins)
    assert isinstance(xbins, np.ndarray)
    assert isinstance(ybins, np.ndarray)
    assert np.array_equal(xbins, expected_x)
    assert np.array_equal(ybins, expected_y)
