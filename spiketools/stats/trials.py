"""Functions for statistically testing trial data."""

import numpy as np
from scipy.stats import ttest_rel, ttest_ind

from spiketools.measures.trials import compute_pre_post_rates, compute_pre_post_averages

###################################################################################################
###################################################################################################

def compute_pre_post_ttest(frs_pre, frs_post):
    """Compute a related samples t-test between the pre & post event firing rates.

    Parameters
    ----------
    frs_pre, frs_post : 1d array
        Firing rates across pre & post event windows.

    Returns
    -------
    t_val : float
        The t-value of the t-test.
    p_val : float
        The p-value of the t-test.

    Examples
    --------
    Compute the t_value & p_value between the firing rates of a 5-minute pre & post event window. 

    >>> frs_pre = np.array([1.5, 1.8, 1.9, 2.0, 2.2])
    >>> frs_post = np.array([5.5, 6.5, 6.6, 6.7, 7.1])
    >>> compute_pre_post_ttest(frs_pre, frs_post)
    (-29.692872320923538, 7.660650245217322e-06)
    """

    t_val, p_val = ttest_rel(frs_pre, frs_post)

    return t_val, p_val


def compare_pre_post_activity(trial_spikes, pre_window, post_window, avg_type='mean'):
    """Compare pre & post activity, computing the average firing rates and a t-test comparison.

    Parameters
    ----------
    trial_spikes : list of 1d array
        Spike times per trial, in seconds.
    pre_window, post_window : list of [float, float]
        The time window to compute firing rate across, for the pre and post event windows.
    avg_type : {'mean', 'median'}
        The type of averaging function to use.

    Returns
    -------
    avg_pre, avg_post : float
        The average firing rates pre and post event.
    t_val, p_val : float
        The t value and p statistic for a t-test comparing pre and post event firing.

    Examples
    --------
    Compute the average firing rates of a 5-minute pre & post event window and a t-test, across 3 trials. 

    >>> trial_spikes = [np.array([0.2, 0.3, 0.4, 0.5, 0.6]), \
                        np.array([0.7, 0.8, 0.9, 1.0, 1.2]), \
                        np.array([1.4, 1.6, 1.7, 1.9, 2.0])]
    >>> pre_window, post_window = [0.1, 1], [1.2, 2]
    >>> compare_pre_post_activity(trial_spikes, pre_window, post_window, avg_type='mean')
    (3.3333333333333335, 2.5, 0.23105423672046252, 0.8387578309724748)
    """

    frs_pre, frs_post = compute_pre_post_rates(trial_spikes, pre_window, post_window)
    avg_pre, avg_post = compute_pre_post_averages(frs_pre, frs_post, avg_type=avg_type)
    t_val, p_val = compute_pre_post_ttest(frs_pre, frs_post)

    return avg_pre, avg_post, t_val, p_val


def compare_trial_frs(trials1, trials2):
    """Compare binned firing rates between two sets of trials with independent samples t-tests.

    Parameters
    ----------
    trials1, trials2 : 2d array
        Precomputed firing rates across bins for two different sets of trials.
        Arrays should be organized as [n_trials, n_bins].

    Returns
    -------
    stats : list of Ttest_indResult
        The statistical results (t-value & p-value) for the t-test, at each bin.
        Output will have the length of n_bins.

    Examples
    --------
    Compare firing rates in a [2, 2] bins between two set of trials that each has 3 trials. 
    
    >>> trials1 = np.array([[1.2, 1.4, 1.6, 1.0], [1.5, 1.9, 0.3, 1.7], [2.1, 1.5, 2.4, 2.2]])
    >>> trials2 = np.array([[4.3, 4.1, 3.9, 4.2], [3.7, 3.4, 3.5, 3.2], [3.9, 4.1, 4.5, 4.7]])
    >>> results = compare_trial_frs(trials1, trials2)
    """

    assert trials1.shape[1] == trials2.shape[1], 'Organization of trials does not line up'

    nbins = trials1.shape[1]
    stats = [ttest_ind(trials1[:, bi], trials2[:, bi]) for bi in range(nbins)]

    return stats
