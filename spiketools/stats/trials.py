"""Functions for statistically testing trial data."""

from scipy.stats import ttest_rel

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
    """

    t_val, p_val = ttest_rel(frs_pre, frs_post)

    return t_val, p_val


def compare_pre_post_activity(trial_spikes, pre_window, post_window, avg_type='mean'):
    """Compare pre & post activity, computing the average firing rates and a t-test comparison.

    Parameters
    ----------
    trial_spikes : list of 1d array
        Spike times per trial.
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
    """

    frs_pre, frs_post = compute_pre_post_rates(trial_spikes, pre_window, post_window)
    avg_pre, avg_post = compute_pre_post_averages(frs_pre, frs_post, avg_type='mean')
    t_val, p_val = compute_pre_post_ttest(frs_pre, frs_post)

    return avg_pre, avg_post, t_val, p_val
