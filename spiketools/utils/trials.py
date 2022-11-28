"""Utility functions for working with trial-level data."""

import numpy as np

###################################################################################################
###################################################################################################

def split_trials_by_condition(trials, conditions):
    """Split trial data by condition label.

    Parameters
    ----------
    trials : list or 2d array
        Trial data.
        If list, each element should represent a trial.
        If array, each row should represent a trial.
    conditions : list
        Condition labels for each trial.

    Returns
    -------
    split_trials : dict
        The trial data, organized by condition.
        Each key is a condition label with values as the data for that condition.
    """

    if isinstance(trials, list):
        out = split_trials_by_condition_list(trials, conditions)
    if isinstance(trials, np.ndarray):
        out = split_trials_by_condition_array(trials, conditions)

    return out

def split_trials_by_condition_list(trials, conditions):
    """Split trial data by condition label, for trial data as a list.

    Parameters
    ----------
    trials : list
        Trial data, with each element representing a trial.
    conditions : list
        Condition labels for each trial.

    Returns
    -------
    split_trials : dict
        The trial data, organized by condition.
        Each key is a condition label with values as the data for that condition.
    """

    msg = 'The number of trials and condition labels does not match.'
    assert len(trials) == len(conditions), msg

    out = {condition : [] for condition in set(conditions)}

    for trial, condition in zip(trials, conditions):
        out[condition].append(trial)

    return out

def split_trials_by_condition_array(trials, conditions):
    """Split trial data by condition label, for trial data as an array.

    Parameters
    ----------
    trials : 2d array
        Trial data, with each row representing a trial.
    conditions : list
        Condition labels for each trial.

    Returns
    -------
    split_trials : dict
        The trial data, organized by condition.
        Each key is a condition label with values as the data for that condition.
    """

    msg = 'The number of trials and condition labels does not match.'
    assert trials.shape[0] == len(conditions), msg

    out = {}
    for condition in set(conditions):
        out[condition] = trials[np.where(np.array(conditions)==condition)[0], :]

    return out
