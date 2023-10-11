"""Simulate spike trains."""

import numpy as np

from spiketools.sim.utils import apply_refractory
from spiketools.utils.checks import check_param_options

###################################################################################################
###################################################################################################

def sim_spiketrain(spike_param, n_samples, method, refractory=None, **kwargs):
    """Simulate a spike train.

    Parameters
    ----------
    spike_param : float
        Parameter value that controls the simulated spiking.
        For `prob` or `binom` methods, this is the probability of spiking.
        For `poisson`, this is the spike rate.
    n_samples : int
        The number of samples to simulate.
    method : {'prob', 'binom', 'poisson'}
        The method to use for the simulation.
    refractory : int, optional
        The refractory period to apply to the simulated data, in number of samples.
    **kwargs
        Additional keyword arguments.
        There are passed into the simulate function specified by `method`.

    Returns
    -------
    spike_train : 1d array
        Simulated spike train.

    Examples
    --------
    Simulate a spike train based on probability of spiking:

    >>> spike_train = sim_spiketrain(0.1, 10, method='prob')

    Simulate a spike train based on a Poisson distribution:

    >>> spike_train = sim_spiketrain(5, 10, method='poisson')
    """

    check_param_options(method, 'method', ['prob', 'binom', 'poisson'])

    spike_train = SPIKETRAIN_FUNCS[method](spike_param, n_samples, **kwargs, refractory=refractory)

    return spike_train

###################################################################################################
## Probability based simulations

@apply_refractory('train')
def sim_spiketrain_prob(p_spiking, n_samples=None, refractory=None):
    """Simulate spikes based on a probability of spiking per sample.

    Parameters
    ----------
    p_spiking : float or 1d array
        The probability (per sample) of spiking.
    n_samples : int, optional
        The number of samples to simulate.
    refractory : int, optional
        The refractory period to apply to the simulated data, in number of samples.

    Returns
    -------
    spike_train : 1d array
        Simulated spike train.

    Raises
    -------
    ValueError
        If the input variable p_spiking is a float and n_samples is None.

    Notes
    -----
    `n_samples` is only used if p_spiking is a float.
    Otherwise `n_samples` is just the length of p_spiking.

    Examples
    --------
    Simulate a spike train based on a probability of spiking per sample:

    >>> p_spiking = 0.3
    >>> n_samples = 10
    >>> sim_spiketrain = sim_spiketrain_prob(p_spiking, n_samples)

    Simulate spike train based on a probability of spiking per sample over time:

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

    spike_train = (probs > np.random.rand(*probs.shape))
    spike_train = spike_train.astype(int)

    return spike_train

###################################################################################################
## Distribution based simulations

@apply_refractory('train')
def sim_spiketrain_binom(p_spiking, n_samples=None, refractory=None):
    """Simulate spike train from a binomial probability distribution.

    Parameters
    ----------
    p_spiking : float or 1d array
        The probability (per sample) of spiking.
    n_samples : int, optional
        The number of samples to simulate.
    refractory : int, optional
        The refractory period to apply to the simulated data, in number of samples.

    Returns
    -------
    spike_train : 1d array
        Simulated spike train.

    Raises
    -------
    ValueError
        If the input variable p_spiking is a float and n_samples is None.

    Notes
    -------
    `n_samples` is only used if p_spiking is a float.
    Otherwise `n_samples` is just the length of `p_spiking`.

    Examples
    --------
    Simulate a spike train based on a probability of spiking per sample:

    >>> p_spiking = 0.7
    >>> spike_train = sim_spiketrain_binom(p_spiking, n_samples=5)

    Simulate spike train with every sample having its own probability of spiking:

    >>> p_spiking = np.array([0, 0.25, 0.5, 0.75, 1])
    >>> spike_train = sim_spiketrain_binom(p_spiking, n_samples=5)
    """

    if isinstance(p_spiking, float) & (n_samples is None):
        raise ValueError("Input variable 'n_samples' must be defined if 'p_spiking' is a float")

    spike_train = np.random.binomial(1, p=p_spiking, size=n_samples)

    return spike_train


@apply_refractory('train')
def sim_spiketrain_poisson(rate, n_samples, fs=1000, refractory=None):
    """Simulate spike train from a Poisson distribution.

    Parameters
    ----------
    rate : float
        The firing rate of neuron to simulate.
    n_samples : int
        The number of samples to simulate.
    fs : int, optional, default: 1000
        The sampling rate, in Hz.
    refractory : int, optional
        The refractory period to apply to the simulated data, in number of samples.

    Returns
    -------
    spike_train : 1d array
        Simulated spike train.

    Examples
    --------
    Simulate a spike train at a rate of 2 Hz for 100 samples:

    >>> spike_train = sim_spiketrain_poisson(2, 100)
    """

    spike_train = np.zeros(n_samples)

    # Create a uniform sampling distribution to use to simulate spikes
    unif = np.random.uniform(0, 1, size=n_samples)
    mask = unif <= (rate * 1 / fs)
    spike_train[mask] = 1

    return spike_train

###################################################################################################
## COLLECT SIM FUNCTION OPTIONS TOGETHER

SPIKETRAIN_FUNCS = {
    'prob' : sim_spiketrain_prob,
    'binom' : sim_spiketrain_binom,
    'poisson' : sim_spiketrain_poisson,
}
