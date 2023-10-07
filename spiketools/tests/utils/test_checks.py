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

def test_check_param_type():

    check_param_type(13, 'test', int)
    check_param_type(1.1, 'test', (int, float))
    check_param_type([1, 2], 'test', [list, np.ndarray])

    with raises(TypeError):
        check_param_type(tuple([1, 2]), 'test', [list, np.ndarray])

def test_check_array_orientation():

    arr1 = np.array([1, 2, 3])
    assert check_array_orientation(arr1) == 'vector'

    arr2r = np.array([[1, 2, 3], [4, 5, 6]])
    assert check_array_orientation(arr2r) == 'row'
    arr2c = np.array([[1, 2], [3, 4], [5, 6]])
    assert check_array_orientation(arr2c) == 'column'

    # Check empty 2d arrays
    arr2re = np.ones((2, 0))
    assert check_array_orientation(arr2re) == 'row'
    arr2ce = np.ones((0, 2))
    assert check_array_orientation(arr2ce) == 'column'

    # Check single sample 2d arrays, with expected shape
    arr2r1s = np.array([[1], [1]])
    assert check_array_orientation(arr2r1s, expected=2) == 'row'
    arr2c1s = np.array([[1, 1]])
    assert check_array_orientation(arr2c1s, expected=2) == 'column'

    # Check 3d arrays
    arr3r = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    assert check_array_orientation(arr3r) == 'row'
    arr3c = np.array([[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]])
    assert check_array_orientation(arr3c) == 'column'

    # Check empty 3d arrays
    arr3re = np.ones((1, 2, 0))
    assert check_array_orientation(arr3re) == 'row'
    arr3ce = np.ones((1, 0, 2))
    assert check_array_orientation(arr3ce) == 'column'

    # Check single sample 3d arrays, with expected shape
    arr3r1s = np.array([[[1], [1]]])
    assert check_array_orientation(arr3r1s, expected=2) == 'row'
    arr3c1s = np.array([[[1, 1]]])
    assert check_array_orientation(arr3c1s, expected=2) == 'column'

def test_check_array_lst_orientation():

    arr_lst1 = [np.array([[], []]), np.array([[1, 2, 4], [5, 6, 7]])]
    out1 = check_array_lst_orientation(arr_lst1)
    assert out1 == 'row'

    arr_lst2 = [np.array([[], []]), np.array([[1, 2, 4], [5, 6, 7]]).T]
    out2 = check_array_lst_orientation(arr_lst2)
    assert out2 == 'column'

    # Check special cases - empty list and no array with enough elements to infer from
    sc_1 = check_array_lst_orientation([])
    assert sc_1 == None
    sc_2 = check_array_lst_orientation([np.array([[1, 2], [3, 4]])])
    assert sc_2 == 'row'

def test_check_axis():

    # Test array inputs
    arr1d = np.array([1, 2, 3])
    arr2dr = np.array([[1, 2, 3], [1, 2, 3]])
    arr2dc = np.array([[1, 2], [1, 2], [1, 2]])

    assert check_axis(0, arr1d) == 0
    assert check_axis(1, arr1d) == 1
    assert check_axis(None, arr1d) == 0
    assert check_axis(None, arr2dr) == 1
    assert check_axis(None, arr2dc) == 0
    assert check_axis(None, arr2dr) == 1

    # Test array list inputs
    arr_lst_1d = [arr1d, arr1d]
    arr_lst_2dr = [arr2dr, arr2dr]
    arr_lst_2dc = [arr2dc, arr2dc]
    arr_lst_emp = []
    assert check_axis(None, arr_lst_1d) == 0
    assert check_axis(None, arr_lst_2dr) == 1
    assert check_axis(None, arr_lst_2dc) == 0
    assert check_axis(None, arr_lst_emp) == -1

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
    out = check_time_bins(tbins, None, tspikes)
    assert np.array_equal(tbins, out)

    # Check time bins given a time resolution, which should create same bins as precomputed
    out = check_time_bins(0.5, [0, 10], tspikes)
    assert np.array_equal(tbins, out)

    # Check error & warning
    with raises(AssertionError):
        out = check_time_bins(np.array([1, 2, 1]), [0, 5], tspikes, True)
    with warns(UserWarning):
        out = check_time_bins(0.5, [0, 5], tspikes, True)

    # Test bin definition with no values provided
    tbins = check_time_bins(0.5, [0, 5])
    assert np.array_equal(tbins, np.arange(0, 5.5, 0.5))
