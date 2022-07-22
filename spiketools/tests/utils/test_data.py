"""Tests for spiketools.utils.data"""

import numpy as np

from spiketools.utils.data import *
from spiketools.utils.data import _include_bin_edge

###################################################################################################
###################################################################################################

def test_compute_range():

    data = np.array([0.5, 1., 1.5, 2., 2.5])
    minv, maxv = compute_range(data)
    assert isinstance(minv, float)
    assert isinstance(maxv, float)
    assert minv == 0.5
    assert maxv == 2.5

def test_smooth_data():

    # Check 1d case
    data = np.array([0.5, 1., 1.5, 2., 2.5])
    out = smooth_data(data, 0.5)
    assert isinstance(out, np.ndarray)
    assert not np.array_equal(data, out)

    # Check 2d case
    data = np.array([[0.5, 1., 1.5, 2., 2.5], [0.5, 1., 1.5, 2., 2.5]])
    out = smooth_data(data, 0.5)
    assert isinstance(out, np.ndarray)
    assert not np.array_equal(data, out)

def test_drop_nans():

    # Check 1d case
    data = np.array([0.5, 1.0, np.nan, 1.5, 2.0, np.nan, 2.5])
    out = drop_nans(data)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, np.array([0.5, 1.0, 1.5, 2.0, 2.5]))

    # Check 2d case
    data = np.array([[0.5, np.nan, 1.0, 1.5, 2.0, np.nan, 2.5],
                     [0.5, np.nan, 1.0, 1.5, 2.0, np.nan, 2.5]])
    out = drop_nans(data)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, np.array([[0.5, 1.0, 1.5, 2.0, 2.5], [0.5, 1.0, 1.5, 2.0, 2.5]]))

def test_assign_data_to_bins():

    data = np.array([1, 3, 5, 7])
    edges = np.array([0, 2, 4, 6, 8])
    assgns = assign_data_to_bins(data, edges)
    expected = np.array([0, 1, 2, 3])
    assert isinstance(assgns, np.ndarray)
    assert np.array_equal(assgns, expected)

def test_include_bin_edge():

    edges = np.array([0, 1, 2])

    # test left side case
    assignments = np.array([1, 2, 3])
    position = np.array([0.5, 1.5, 2])
    out = _include_bin_edge(assignments, position, edges, side='left')
    assert np.array_equal(out, np.array([1, 2, 2]))

    # test right right case
    assignments = np.array([0, 1, 2])
    position = np.array([0, 0.5, 1.5])
    out = _include_bin_edge(assignments, position, edges, side='right')
    assert np.array_equal(out, np.array([1, 1, 2]))

