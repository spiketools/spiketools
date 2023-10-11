"""Simulate trials - 2D arrays of event related spiking activity."""

import numpy as np

from spiketools.sim.times import sim_spiketimes_poisson
from spiketools.utils.checks import check_param_options

###################################################################################################
###################################################################################################

def sim_trials(n_trials, method=None, time_pre=1, time_post=2, refractory=0.001, **kwargs):
    """Simulate a collection of trials.

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.
    method : {'poisson'}
        The method to use for the simulation.
    time_pre, time_post : float
        The amount of time to simulate pre / post the event.
    refractory : float, optional, default: 0.001
        The refractory period to apply to the simulated data, in seconds.
    **kwargs
        Additional keyword arguments.
        There are passed into the simulate function specified by `method`.

    Returns
    -------
    trial_spikes : list of 1d array
        Simulated trials, where list has length of n_trials.
        Each simulated trial has simulated spike times for the time range [-time_pre, time_post].
    """

    check_param_options(method, 'method', ['poisson'])

    trial_spikes = SPIKETRIAL_FUNCS[method](\
        n_trials, **kwargs, time_pre=time_pre, time_post=time_post, refractory=refractory)

    return trial_spikes


###################################################################################################
## Distribution based simulations

def sim_trials_poisson(n_trials, rate_pre, rate_post, time_pre=1, time_post=2, refractory=0.001):
    """Simulate a collection of trials, based on a Poisson spike time simulation.

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.
    rate_pre, rate_post : float
        The firing rates for the pre and post event times.
    time_pre, time_post : float
        The amount of time to simulate pre / post the event.
    refractory : float, optional, default: 0.001
        The refractory period to apply to the simulated data, in seconds.

    Returns
    -------
    trial_spikes : list of 1d array
        Simulated trials, where list has length of n_trials.
        Each simulated trial has simulated spike times for the time range [-time_pre, time_post].
    """

    trial_spikes = [None] * n_trials
    for trial_idx in range(n_trials):

        trial_spikes[trial_idx] = np.append(
            sim_spiketimes_poisson(rate_pre, time_pre, start_time=-time_pre, refractory=refractory),
            sim_spiketimes_poisson(rate_post, time_post, start_time=0, refractory=refractory),
        )

    return trial_spikes

###################################################################################################
## COLLECT SIM FUNCTION OPTIONS TOGETHER

SPIKETRIAL_FUNCS = {
    'poisson' : sim_trials_poisson,
}
