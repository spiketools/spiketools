"""Functions to compute trial-related measures."""

import numpy as np

from spiketools.utils.checks import check_time_bins
from spiketools.measures.conversions import convert_times_to_rates

###################################################################################################
###################################################################################################

def compute_trial_frs(trial_spikes, bins, trange=None, smooth=None):
    """Compute continuous binned firing rates for a set of epoched spike times.

    Parameters
    ----------
    trial_spikes : list of 1d array
        Spike times per trial.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    trange : list of [float, float]
        Time range, in seconds, to create the binned firing rate across.
        Only used if `bins` is a float.
    smooth : float, optional
        If provided, the kernel to use to smooth the continuous firing rate.

    Returns
    -------
    trial_cfrs : 2d array
        Continuous firing rates per trial, with shape [n_trials, n_time_bins].
    """

    bins = check_time_bins(bins, trial_spikes[0], trange=trange)
    trial_cfrs = np.zeros([len(trial_spikes), len(bins) - 1])
    for ind, t_spikes in enumerate(trial_spikes):
        trial_cfrs[ind, :] = convert_times_to_rates(t_spikes, bins, smooth)

    return trial_cfrs
