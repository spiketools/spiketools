"""Tests for spiketools.utils.data"""

import numpy as np

from spiketools.utils.data import *
from spiketools.utils.data import _include_bin_edge

###################################################################################################
###################################################################################################

def test_make_orientation():

    arr_r = np.array([[1, 2, 3], [4, 5, 6]])
    arr_c = np.array([[1, 4], [2, 5], [3, 6]])

    out_rr = make_orientation(arr_r, 'row')
    assert np.array_equal(out_rr, arr_r)

    out_cr = make_orientation(arr_c, 'row')
    assert np.array_equal(out_cr, arr_r)

    out_rc = make_orientation(arr_r, 'column')
    assert np.array_equal(out_rc, arr_c)

    out_cc = make_orientation(arr_c, 'column')
    assert np.array_equal(out_cc, arr_c)

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

def test_permute_vector():

    n_permutations = 5
    data = np.array([1, 2, 3, 4, 5])

    out = permute_vector(data, n_permutations=n_permutations)
    assert isinstance(out, np.ndarray)
    assert out.shape == (n_permutations, len(data))

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

