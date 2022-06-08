"""Simulate spike trains."""

import numpy as np

from spiketools.sim.utils import refractory

###################################################################################################
###################################################################################################

def sim_spiketrain(spike_param, n_samples, method, **kwargs):
    """Simulate a spike train.

    Parameters
    ----------
    spike_param : float
        Parameter value that controls the simulated spiking. rate or probability.
        For `prob` or `binom` methods, this is the probability of spiking.
        For `poisson`, this is the spike rate.
    n_samples : int
        The number of samples to simulate.
    method : {'prob', 'binom', 'poisson'}
        The method to use for the simulation.
    **kwargs
        Additional keyword arguments

    Returns
    -------
    train : 1d array
        Simulated spike train.
    """

    func = SPIKETRAIN_FUNCS[method]

    train = func(spike_param, n_samples, **kwargs)

    return train

###################################################################################################
## Probability based simulations

def sim_spiketrain_prob(p_spiking, n_samples=None):
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
    -----
    n_samples is only used if p_spiking is a float.
    Otherwise n_samples is just the length of p_spiking.

    Examples
    --------
    Simulate spike train of size n_samples, based on a probability of spiking per sample.

    >>> p_spiking = 0.3
    >>> n_samples = 10
    >>> sim_spiketrain = sim_spiketrain_prob(p_spiking, n_samples)

    Simulate spike train based on a probability of spiking per sample over time.

    >>> p_spiking = np.array([0.3, 0.5, 0.6])
    >>> sim_spiketrain = sim_spiketrain_prob(p_spiking)
    """

    if isinstance(p_spiking, float):
        if n_samples is not None:
            probs = (np.ones(n_samples) * p_spiking)
        else:
            msg = "Input variable 'n_samples' must be defined if 'p_spiking' is a float"
            raise ValueError(msg)
    else:
        probs = p_spiking

    spikes = (probs > np.random.rand(*probs.shape))
    spikes = spikes.astype(int)

    return spikes


###################################################################################################
## Distribution based simulations

def sim_spiketrain_binom(p_spiking, n_samples=None):
    """Simulate spike train from a binomial probability distribution.

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
    n_samples is only used if p_spiking is a float.
    Otherwise n_samples is just the length of p_spiking.

    Examples
    --------
    Simulate spike train with every sample having the same probabiliy of spiking:

    >>> p_spiking = 0.7
    >>> spikes = sim_spiketrain_binom(p_spiking, n_samples=5)

    Simulate spike train with every sample having its own probability of spiking:

    >>> p_spiking = np.array([0, 0.25, 0.5, 0.75, 1])
    >>> spikes = sim_spiketrain_binom(p_spiking, n_samples=5)
    """

    if isinstance(p_spiking, float) & (n_samples is None):
        raise ValueError("Input variable 'n_samples' must be defined if 'p_spiking' is a float")

    return np.random.binomial(1, p=p_spiking, size=n_samples)


def sim_spiketrain_poisson(rate, n_samples, fs=1000, bias=0):
    """Simulate spike train from a Poisson distribution.

    Parameters
    ----------
    rate : float
        The firing rate of neuron to simulate.
    n_samples : int
        The number of samples to simulate.
    fs : int
        The sampling rate.

    Returns
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

###################################################################################################
## COLLECT SIM FUNCTION OPTIONS TOGETHER

SPIKETRAIN_FUNCS = {'prob' : sim_spiketrain_prob,
                    'binom' : sim_spiketrain_binom,
                    'poisson' : sim_spiketrain_poisson}
