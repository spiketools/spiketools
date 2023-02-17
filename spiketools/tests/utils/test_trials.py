"""Tests for spiketools.utils.trials"""

import numpy as np

from spiketools.utils.trials import *

###################################################################################################
###################################################################################################

def test_split_trials_by_condition():

    trials_lst = [[1, 2, 3], [1, 3], [1, 4, 5]]
    trials_arr = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
    conditions = ['A', 'B', 'A']

    # Note that accuracy checking done in sub-functions - this just checks execution
    out_lst = split_trials_by_condition(trials_lst, conditions)
    assert out_lst
    out_arr = split_trials_by_condition(trials_arr, conditions)
    assert out_arr

def test_split_trials_by_condition_list():

    trials = [[1, 2, 3], [1, 3], [1, 4, 5]]
    conditions = ['A', 'B', 'A']

    out = split_trials_by_condition_list(trials, conditions)

    assert out['A'] == [[1, 2, 3], [1, 4, 5]]
    assert out['B'] == [[1, 3]]

def test_split_trials_by_condition_array():

    trials = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
    conditions = ['A', 'B', 'A']

    out = split_trials_by_condition_list(trials, conditions)

    assert np.array_equal(out['A'], np.array([[1, 2, 3], [4, 5, 6]]))
    assert np.array_equal(out['B'], np.array([[2, 3, 4]]))

def test_recombine_trial_data():

    trial_times = [np.array([1, 2, 3]), np.array([6, 7])]
    expected_times = np.array([1, 2, 3, 6, 7])

    # 1d test
    trial_values = [np.array([10, 11, 12]), np.array([1, 2])]
    times_1d, values_1d = recombine_trial_data(trial_times, trial_values)
    assert np.array_equal(times_1d, expected_times)
    assert np.array_equal(values_1d, np.array([10, 11, 12, 1, 2]))

    # 2d test (row data)
    trial_values_2dr = [np.array([[10, 11, 12], [20, 21, 22]]), np.array([[1, 2], [1, 2]])]
    times_2dr, values_2dr = recombine_trial_data(trial_times, trial_values_2dr)
    assert np.array_equal(times_2dr, expected_times)
    assert np.array_equal(values_2dr, np.array([[10, 11, 12, 1, 2], [20, 21, 22, 1, 2]]))

    # 2d test (column data)
    trial_values_2dc = [el.T for el in trial_values_2dr]
    times_2dc, values_2dc = recombine_trial_data(trial_times, trial_values_2dc)
    assert np.array_equal(times_2dc, expected_times)
    assert np.array_equal(values_2dc, np.array([[10, 11, 12, 1, 2], [20, 21, 22, 1, 2]]).T)

    # test with empty trials
    trial_times_missing = [np.array([]), np.array([1, 2, 3]), np.array([]), np.array([6, 7])]
    trial_values_missing = [np.array([]), np.array([10, 11, 12]), np.array([]), np.array([1, 2])]
    times_missing, values_missing = recombine_trial_data(trial_times_missing, trial_values_missing)
    assert np.array_equal(times_missing, expected_times)
    assert np.array_equal(values_missing, np.array([10, 11, 12, 1, 2]))
