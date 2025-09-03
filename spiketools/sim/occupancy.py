"""Simulate occupancy."""

import numpy as np

###################################################################################################
###################################################################################################

def sim_single_trial_occ(num_bins, min_occ, max_occ):
    """Simulate occupancy for a single trial.

    Parameters
    ----------
    num_bins: int
        Number of spatial bins.
    min_occ: float
        Minimum occupancy.
    max_occ: float 
        Maximum occupancy.

    Returns
    -------
    occupancy: array-like
        Occupancy for a single trial.
    """

    return np.random.uniform(min_occ, max_occ, size=num_bins)  


def sim_single_uniform_occ(num_bins, scale):
    """Simulate uniform occupancy for a single trial.
    Parameters
    ----------
    num_bins: int
        Number of spatial bins.
    scale: float
        Scale of the occupancy.

    Returns
    -------
    occupancy: array-like
        Occupancy for a single trial.
    """

    return np.ones(num_bins) * scale


def sim_occ_trials(num_trials, num_bins, min_occ, max_occ, use_random=False, uniform_scale=1):
    """Simulate occupancy for multiple trials.

    Parameters
    ----------
    num_trials: int
        Number of trials.
    num_bins: int
        Number of spatial bins.
    min_occ: float
        Minimum occupancy.
    max_occ: float
        Maximum occupancy.
    use_random: bool
        Whether to use random occupancy.
    uniform_scale: float
        Scale of the occupancy.

    Returns
    -------
    all_trials_occ: array-like
        Occupancy for all trials.
    avg_occ: array-like
        Average occupancy.
    """

    all_trials_occ=[]
    for _ in range(num_trials):
        if use_random:
            trial_occ=np.random.uniform(min_occ, max_occ, size=num_bins)
        else:
            trial_occ=np.ones(num_bins)*uniform_scale
        all_trials_occ.append(trial_occ)

    all_trials_occ = np.array(all_trials_occ)
    avg_occ= np.sum(all_trials_occ, axis=0)/num_trials
    return all_trials_occ, avg_occ
