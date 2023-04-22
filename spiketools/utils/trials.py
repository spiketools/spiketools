"""Utility functions for working with trial-level data."""

import numpy as np

from spiketools.utils.checks import check_param_lengths, check_param_type, check_axis

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

    check_param_type(trials, 'trials', (list, np.ndarray))
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


def recombine_trial_data(times_trials, values_trials, axis=None):
    """Recombine data across trials.

    Parameters
    ----------
    times_trials : list of 1d array
        Epoched timestamps.
    values_trials : list of 1d or 2d array
        Epoched trial data.
    axis : {0, 1}, optional
        The axis argument for the `values` data, if it's a 2d array, as {0: column, 1: row}.
        If not provided, is inferred from the `values` array.

    Returns
    -------
    time : 1d array
        Recombined timestamps.
    values : 1d or 2d array
        Recombined trial data.
    """

    check_param_lengths([times_trials, values_trials], ['times_trials', 'values_trials'])

    times = np.concatenate(times_trials)
    values = np.concatenate(values_trials, axis=check_axis(axis, values_trials[0]))

    return times, values
