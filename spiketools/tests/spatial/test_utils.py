"""Tests for spiketools.spatial.utils"""

import numpy as np

from spiketools.spatial.utils import *

###################################################################################################
###################################################################################################

def test_get_position_xy():

    position_2dr = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
    position_2dc = position_2dr.T

    expected_x = np.array([1, 2, 3, 4, 5])
    expected_y = np.array([5, 6, 7, 8, 9])

    # Test 2d position data (rows)
    x_data, y_data = get_position_xy(position_2dr)
    assert np.array_equal(x_data, expected_x)
    assert np.array_equal(y_data, expected_y)

    # Test 2d position data (columns)
    x_data, y_data = get_position_xy(position_2dc)
    assert np.array_equal(x_data, expected_x)
    assert np.array_equal(y_data, expected_y)

    # Test single data value cases
    x_r1s, y_r1s = get_position_xy(np.array([[1], [5]]))
    assert np.array_equal(x_r1s, np.array([1]))
    assert np.array_equal(y_r1s, np.array([5]))
    x_c1s, y_c1s = get_position_xy(np.array([[1, 5]]))
    assert np.array_equal(x_c1s, np.array([1]))
    assert np.array_equal(y_c1s, np.array([5]))

def test_compute_nbins():

    # check 1d case
    out1 = compute_nbins([5, 5])
    assert out1 == 25

    # check 2d case
    out2 = compute_nbins([5])
    out3 = compute_nbins(5)
    assert out2 == out3 == 5

def test_compute_pos_ranges():

    # Test 1d position data
    positions = np.array([1, 2, 3, 4, 5])
    ranges = compute_pos_ranges(positions)
    assert np.array_equal(ranges, np.array([1, 5]))

    # 2d tests
    positions = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
    expected_x = np.array([1, 5])
    expected_y = np.array([5, 9])

    # Test 2d position data (row data)
    ranges_2d = compute_pos_ranges(positions)
    assert np.array_equal(ranges_2d[0], expected_x)
    assert np.array_equal(ranges_2d[1], expected_y)

    # Test 2d position data (column data)
    ranges_2dc = compute_pos_ranges(positions.T)
    assert np.array_equal(ranges_2dc[0], expected_x)
    assert np.array_equal(ranges_2dc[1], expected_y)

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
