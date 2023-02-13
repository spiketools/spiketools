"""Tests for spiketools.utils.checks"""

import numpy as np

from pytest import raises, warns

from spiketools.utils.checks import *

###################################################################################################
###################################################################################################

def test_check_param_range():

    # Check that valid options run without error
    check_param_range(0.5, 'test', [0., 1])
    check_param_range(0., 'test', [0., 1])
    check_param_range(1., 'test', [0., 1])
    check_param_range('a', 'test', ['a', 'b'])

    # Check that invalid options raise an error
    with raises(ValueError):
        check_param_range(-1, 'test', [0., 1])
    with raises(ValueError):
        check_param_range(1.5, 'test', [0., 1])

def test_check_param_options():

    # Check that valid options run without error
    check_param_options('a', 'test', ['a', 'b', 'c'])

    with raises(ValueError):
        check_param_options('a', 'test', ['b', 'c'])

def test_check_list_options():

    check_list_options(['a', 'b', 'c'], 'test', ['a', 'b', 'c'])

    with raises(ValueError):
        check_list_options(['a', 'b', 'c'], 'test', ['b', 'c'])

def test_check_param_lengths():

    a1 = [1, 2, 3]
    a2 = ['a', 'b', 'c']
    a3 = [True, False]

    check_param_lengths([a1, a2], ['a1', 'a2'])

    with raises(ValueError):
        check_param_lengths([a1, a3], ['a1', 'a3'])

    check_param_lengths([a1, a2], ['a1', 'a2'], expected_length=3)
    with raises(ValueError):
        check_param_lengths([a1, a2], ['a1', 'a2'], expected_length=2)

def test_check_array_orientation():

    arr1 = np.array([1, 2, 3])
    assert check_array_orientation(arr1) == 'vector'

    arr2r = np.array([[1, 2, 3], [4, 5, 6]])
    assert check_array_orientation(arr2r) == 'row'
    arr2c = np.array([[1, 2], [3, 4], [5, 6]])
    assert check_array_orientation(arr2c) == 'column'

    arr3r = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    assert check_array_orientation(arr3r) == 'row'
    arr3c = np.array([[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]])
    assert check_array_orientation(arr3c) == 'column'

def test_check_bin_range():

    values = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    edges1 = np.array([0, 3, 6])
    edges2 = np.array([1, 2.5, 5])
    edges3 = np.array([0, 2, 4])

    check_bin_range(values, edges1)
    with warns(UserWarning):
        check_bin_range(values, edges2)
    with warns(UserWarning):
        check_bin_range(values, edges3)

def test_check_time_bins(tspikes):

    # Check precomputed time bins
    tbins = np.arange(0, 10 + 0.5, 0.5)
    out = check_time_bins(tbins, tspikes)
    assert np.array_equal(tbins, out)

    # Check time bins given a time resolution, which should create same bins as precomputed
    out = check_time_bins(0.5, tspikes, time_range=[0, 10])
    assert np.array_equal(tbins, out)

    # Check error & warning
    with raises(AssertionError):
        out = check_time_bins(np.array([1, 2, 1]), tspikes, time_range=[0, 5])
    with warns(UserWarning):
        out = check_time_bins(0.5, tspikes, time_range=[0, 5])

    # Test bin definition with no values provided
    tbins = check_time_bins(0.5, None, [0, 5])
    assert np.array_equal(tbins, np.arange(0, 5.5, 0.5))
