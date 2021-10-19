"""Simulate spikes based on spiking probabilities."""

import numpy as np

###################################################################################################
###################################################################################################

def sim_spiketrain_prob(p_spiking, n_samples):
    """Simulate spikes based on a probability of spiking per unit.

    Parameters
    ----------
    p_spiking : float or 1d array
        The probability (per unit) of spiking.
    n_samples : int, optional
        The number of samples per unit to simulate.

    Returns
    -------
    spikes : nd array
        Simulated spike train per unit.
		
	Notes
	-------
	When p_spiking is an array, each unit corresponds to a row in spikes.
    """
    if isinstance(p_spiking, float):
		probs = np.ones(n_samples) * p_spiking
    else:
		# for many samples being simulated, each unit corresponds to one row
		probs = np.ones((len(p_spiking), n_samples))* (p_spiking[:, np.newaxis])

	spikes = (probs > np.random.rand(*probs.shape))
	spikes = spikes.astype(int)
	
	return spikes
