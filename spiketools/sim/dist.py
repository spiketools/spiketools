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
    
    Raises
    -------
    ValueError
        If the input variable p_spiking is a float and n_samples is None.
    
    Notes
    -------
    n_samples is only used if p_spiking is a float, otherwise n_samples is just the length of p_spiking.
    
    Examples
    --------
    Simulate spike train from a binomial probability distribution (every sample shares the same probabiliy of spiking).
    
    >>> p_spiking = 0.7
    >>> spikes = sim_spiketrain_binom(p_spiking, n_samples=5)
    
    Simulate spike train from a binomial probability distribution (every sample has its own probability of spiking).
    
    >>> p_spiking = np.array([0, 0.25, 0.5, 0.75, 1])
    >>> spikes = sim_spiketrain_binom(p_spiking, n_samples=5)
    """

    if isinstance(p_spiking, float) & (n_samples == None):
        raise ValueError("Input variable 'n_samples' must be defined if 'p_spiking' is a float")
    else:
        return np.random.binomial(1, p=p_spiking, size=n_samples)
    
    
def sim_spiketrain_poisson(rate, n_samples, fs, bias=0):
    """Simulate spike train from a Poisson distribution.
    
    Parameters
    ----------
    rate : float
        The firing rate of neuron to simulate.
    n_samples : int
        The number of samples to simulate.
    fs : int
        The sampling rate.
    
    Results
    -------
    spikes : 1d array
        Simulated spike train.
        
    Examples
    --------
    Simulate a spike train from a Poisson distribution.
    
    >>> spikes = sim_spiketrain_poisson(0.4, 10, 1000, bias=0)
    """

    spikes = np.zeros(n_samples)

    # Create uniform sampling distribution
    unif = np.random.uniform(0, 1, size=n_samples)

    # Create spikes
    mask = unif <= ((rate + bias) * 1/fs)
    spikes[mask] = 1

    return spikes
