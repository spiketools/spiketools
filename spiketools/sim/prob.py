"""Simulate spikes based on spiking probabilities."""

import numpy as np

###################################################################################################
###################################################################################################

def sim_spiketrain_prob(p_spiking, n_samples):
    """Simulate spikes based on a probability of spiking per sample.

    Parameters
    ----------
    p_spiking : float or 1d array
            The probability (per sample) of spiking.
    n_samples : int, optional
        The number of samples to simulate.

    Returns
    -------
    spikes : 1d array
        Simulated spike train.
    """

    if isinstance(prob, float):
        probs = np.ones(length) * prob

    spikes = (probs > np.random.rand(*probs.shape))
    spikes = spikes.astype(int)

    return spikes
