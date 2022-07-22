"""Simulate spike times."""

import numpy as np

from spiketools.stats.generators import poisson_generator
from spiketools.sim.utils import apply_refractory_times
from spiketools.utils.checks import check_param_options

###################################################################################################
###################################################################################################

def sim_spiketimes(spike_param, duration, method, refractory=None, **kwargs):
    """Simulate spike times.

    Parameters
    ----------
    spike_param : float
        Parameter value that controls the simulated spiking. rate or probability.
        For `poisson`, this is the spike rate.
    duration : float
        Duration of spike times to simulate.
    method : {'poisson'}
        The method to use for the simulation.
    refractory : float, optional
        The refractory period to apply to the simulated data, in seconds.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    times : 1d array
        Simulated spike times, in seconds.

    Examples
    --------
    Simulate 10 spike times at 5Hz for 3 seconds, with possion distribution. 

    >>> spikes = sim_spiketimes(5, 3, 'poisson')
    """

    check_param_options(method, 'method', ['poisson'])

    times = SPIKETIME_FUNCS[method](spike_param, duration, **kwargs)

    if refractory:
        times = apply_refractory_times(times, refractory)

    return times

###################################################################################################
## Distribution based simulations

def sim_spiketimes_poisson(rate, duration, start_time=0):
    """Simulate spike times based on a Poisson distribution.

    Parameters
    ----------
    rate : float
        The average firing rate for the simulated spike times.
    duration : float
        Duration of spike times to simulate, in seconds.
    start_time: float, optional
        Timestamp of the start time for the simulated spike times.

    Returns
    -------
    times : 1d array
        Simulated spike times, in seconds.

    Examples
    --------
    Simulate spike times at 10Hz for 5 seconds, starting at 2 seconds, with poisson distribution. 

    >>> spikes = sim_spiketimes_poisson(10, 5, start_time=2)
    """

    times = np.array(list(poisson_generator(rate, duration, start_time)))

    return times

###################################################################################################
## COLLECT SIM FUNCTION OPTIONS TOGETHER

SPIKETIME_FUNCS = {'poisson' : sim_spiketimes_poisson}
