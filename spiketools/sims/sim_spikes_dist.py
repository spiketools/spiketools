"""Simulate spikes based on different probability distributions."""

import numpy as np

###################################################################################################
###################################################################################################

def sim_spiketrain_binom(p_spiking, n_samples=None):
    """Simulate spike train from a binomial probability distribution.

    Parameters
    ----------
    p_spiking : float or 1d array
        The probability (per sample) of spiking.
    n_samples : int, optional
        The number of samples to simulate.

    Results
    -------
    spikes : 1d array
        Simulated spike train.
    """

    return np.random.binomial(1, p=p_spiking, size=n_samples)
