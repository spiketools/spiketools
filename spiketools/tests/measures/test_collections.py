"""Tests for spiketools.measures.all"""

import numpy as np

from spiketools.measures.collections import *

###################################################################################################
###################################################################################################

def test_detect_empty_time_ranges():

    all_spikes = [np.array([0.5, 1.1, 1.9, 4.1]),
                  np.array([1.5, 2.5, 4.25])]
    empty_ranges = detect_empty_time_ranges(all_spikes, 1, [0, 5])
    assert empty_ranges == [[3, 4]]


def test_find_empty_bins():

    all_spikes = [np.array([0.5, 1.1, 1.9, 4.1]),
                  np.array([1.5, 2.5, 4.25])]

    empty_bins = find_empty_bins(all_spikes, 1, [0, 5])
    assert isinstance(empty_bins, np.ndarray)
    assert empty_bins.dtype == 'bool'
    assert np.array_equal(empty_bins, np.array([False, False, False, True, False]))

def test_find_empty_ranges():

    empty_bins = np.array([False, False, False, True, False])
    empty_ranges = find_empty_ranges(empty_bins, 1, [0, 5])
    assert empty_ranges == [[3, 4]]

    empty_bins = np.array([False, True, True, False, False, False, True, True, False, False])
    empty_ranges = find_empty_ranges(empty_bins, 0.5, [0, 5])
    assert empty_ranges == [[0.5, 1.5], [3.0, 4.0]]

    # Test case with empty ranges at beginning and end
    empty_bins = np.array([True, True, False, False, False, True, True, False, False, True])
    empty_ranges = find_empty_ranges(empty_bins, 0.5, [0, 5])
    assert empty_ranges == [[0., 1.], [2.5, 3.5], [4.5, 5]]
