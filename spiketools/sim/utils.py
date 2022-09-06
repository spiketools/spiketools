"""Utilities for working with simulated spiking data."""

from functools import wraps

import numpy as np

from spiketools.modutils.functions import get_function_argument

###################################################################################################
###################################################################################################

def apply_refractory_times(spike_times, refractory_time):
    """Apply a refractory period to simulated spike times.

    Parameters
    ----------
    spike_times : 1d array
        Spike times.
    refractory_time : float
        The duration of the refractory period, after a spike, in seconds.

    Returns
    -------
    spike_times : 1d array
        Spike times, with refractory period applied.

    Examples
    --------
    Apply a 0.003 seconds refractory period to a set of spike times:

    >>> spike_times = np.array([0.512, 1.241, 1.242, 1.751, 2.124])
    >>> apply_refractory_times(spike_times, 0.003)
    array([0.512, 1.241, 1.751, 2.124])
    """

    mask = np.diff(spike_times) > refractory_time
    mask = np.insert(mask, 0, True)

    spike_times = spike_times[mask]

    return spike_times


def refractory_times(func):
    """Decorator for applying a refractory period to spike time simulations."""

    @wraps(func)
    def decorated(*args, **kwargs):

        refractory = get_function_argument('refractory', func, args, kwargs)

        spike_times = func(*args, **kwargs)

        if refractory:
            spike_times = apply_refractory_times(spike_times, refractory)

        return spike_times

    return decorated


def apply_refractory_train(spike_train, refractory_time, fs=1000):
    """Apply a refractory period to a simulated spike train.

    Parameters
    ----------
    spike_train : 1d array
        Spike train.
    refractory_time : float
        The duration of the refractory period, after a spike, in seconds.
    fs : float, optional, default: 1000
        The sampling rate of the spike train.

    Returns
    -------
    spike_train : 1d array
        Spike train, with refractory period applied.

    Examples
    --------
    Apply a 0.003 seconds refractory period to a binary spike train with 1000 Hz sampling rate:

    >>> spike_train = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0])
    >>> apply_refractory_train(spike_train, 0.003, 1000)
    array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    """

    ref_len = int(refractory_time * fs)

    for ind in range(spike_train.shape[0]):
        if spike_train[ind]:
            spike_train[ind+1:ind+ref_len] = 0

    return spike_train
