"""Utilities for managing trials and epochs."""

from spiketools.utils.data import restrict_range

###################################################################################################
###################################################################################################

def epoch_trials(spikes, events, window):
    """Epoch spiking data into trials.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    events : 1d array
        The set of event times to extract from the data.
    window : list of [float, float]
        The time window to extract around each event.

    Returns
    -------
    trials : list of 1d array
        Spike

    Notes
    -----
    For each trial, the returned spike times will be relative to each event time, set as zero.
    """

    trials = [None] * len(events)
    for ind, event in enumerate(events):
        trials[ind] = restrict_range(spikes, event + window[0], event + window[1]) - event

    return trials
