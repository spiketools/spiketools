"""Utilities for working with simulated spiking data."""

###################################################################################################
###################################################################################################

def refractory(spikes, refractory_time, fs):
    """Apply a refractory period to a simulated spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike train.
    refractory_time : float
        The duration of the refractory period, after a spike, in seconds.
    fs : float
        The sampling rate.

    Returns
    -------
    spikes : 1d array
        Spike train, with refractory period constraint applied.
    """

    ref_len = int(refractory_time * fs)

    for ind in range(spikes.shape[0]):
        if spikes[ind]:
            spikes[ind+1:ind+ref_len] = 0

    return spikes
