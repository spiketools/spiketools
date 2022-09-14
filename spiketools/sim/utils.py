"""Utilities for working with simulated spiking data."""

from functools import wraps

import numpy as np

from spiketools.utils.checks import check_param_options
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


def apply_refractory_train(spike_train, refractory_samples):
    """Apply a refractory period to a simulated spike train.

    Parameters
    ----------
    spike_train : 1d array
        Spike train.
    refractory_samples : int
        The duration of the refractory period, after a spike, in number of samples.

    Returns
    -------
    spike_train : 1d array
        Spike train, with refractory period applied.

    Examples
    --------
    Apply a 1-sample refractory period to a spike train:

    >>> spike_train = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
    >>> apply_refractory_train(spike_train, 1)
    array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    """

    for ind in range(spike_train.shape[0]):
        if spike_train[ind]:
            spike_train[ind + 1:ind + 1 + refractory_samples] = 0

    return spike_train

###################################################################################################
## COLLECT REFRACTORY FUNCTION OPTIONS TOGETHER

REFRACTORY_FUNCTIONS = {
    'times' : apply_refractory_times,
    'train' : apply_refractory_train,
}

###################################################################################################
## REFRACTORY DECORATOR

def apply_refractory(spike_representation):
    """Decorator for applying a refractory period to simulated spiking data.

    Parameters
    ----------
    spike_representation : {'times', 'train'}
        Defines the representation of the simulated spikes.
        Based on this input, the decorator applies the appropriate refractory function.

    Notes
    -----
    This decorator manages applying a refractory period to simulated spiking data.
    It assumes the following aspects:
    - the wrapped function takes `refractory` as an argument, in the last position
        - the expected units of this argument vary by `spike_representation`
            - for `times`, refractory should be in seconds
            - for `train`, refractory should be in samples
        - if `refractory` is defined (not None), as a specified input or as
          as default value in the function signature, this refractory period is applied
    - the wrapped function outputs a single `spikes` output
    """

    def wrap(func):

        check_param_options(spike_representation, 'spike_representation', ['times', 'train'])
        refractory_function = REFRACTORY_FUNCTIONS[spike_representation]

        @wraps(func)
        def decorated(*args, **kwargs):

            refractory = get_function_argument('refractory', func, args, kwargs)

            spikes = func(*args, **kwargs)

            if refractory:
                spikes = refractory_function(spikes, refractory)

            return spikes
        return decorated
    return wrap
