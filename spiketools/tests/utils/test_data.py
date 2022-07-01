"""Tests for spiketools.utils.data"""

import numpy as np

from spiketools.utils.data import *

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
