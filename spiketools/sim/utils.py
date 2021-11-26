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

    Examples
    --------
    Apply a 0.003 seconds refractory period to a binary spike train with 1000 Hz sampling rate.

    >>> spikes = np.array([0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,0,])
    >>> refractory(spikes, 0.003, 1000)
    array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    """

    ref_len = int(refractory_time * fs)

    for ind in range(spikes.shape[0]):
        if spikes[ind]:
            spikes[ind+1:ind+ref_len] = 0

    return spikes
