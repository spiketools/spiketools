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
