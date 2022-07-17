"""Tests for spiketools.spatial.target"""

import numpy as np

from spiketools.spatial.target import *

###################################################################################################
###################################################################################################

def test_compute_target_bins():

    target_frs = np.array([[2.0, 1.0, 2.0], [1.0, 2.0, 1.0]])
    xbins = np.array([0, 1, 2, 0, 1, 2])
    ybins = np.array([0, 1, 0, 1, 0, 1])

    # test 1d case
    bins1d = [3]
    target_bins = compute_target_bins(target_frs, bins1d, xbins)
    assert isinstance(target_bins, np.ndarray)
    assert np.array_equal(target_bins.shape, np.array(bins1d))
    expected = np.array([3.0, 3.0, 3.0])
    assert np.array_equal(target_bins, expected)

    # test 2d case
    bins2d = [3, 2]
    target_bins = compute_target_bins(target_frs, bins2d, xbins, ybins)
    assert isinstance(target_bins, np.ndarray)
    assert np.array_equal(target_bins.shape, np.flip(bins2d))
    expected = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    assert np.array_equal(target_bins, expected)

    # test with occupancy normalization
    target_occupancy = np.array([[2, 2, 2], [1, 1, 1]])
    target_bins = compute_target_bins(target_frs, bins2d, xbins, ybins, target_occupancy)
    expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert np.array_equal(target_bins, expected)
