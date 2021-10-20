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

    Raises
    -------
    ValueError
        If the input variable p_spiking is a float and n_samples is None.
		
    Notes
    -------
    n_samples is only used if p_spiking is a float, otherwise n_samples is just the length of p_spiking.
    """

    if isinstance(p_spiking, float):
        if n_samples != None:
            probs = (np.ones(n_samples) * p_spiking)
        else:
            raise ValueError("Input variable 'n_samples' must be defined if p_spiking is a float")
    else:
        probs = p_spiking

    spikes = (probs > np.random.rand(*probs.shape))
    spikes = spikes.astype(int)

    return spikes
