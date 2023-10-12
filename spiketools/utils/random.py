"""Random state related functions."""

import numpy as np

###################################################################################################
###################################################################################################

def set_random_seed(seed_value=0):
    """Set the random seed value.

    Parameters
    ----------
    seed_value : int, optional
        Value to set the random seed as.

    Notes
    -----
    This sets the random state of the global numpy state.
    """

    np.random.seed(seed_value)


def set_random_state(seed_value=0):
    """Set the random state in a RandomState object.

    Parameters
    ----------
    seed_value : int, optional
        Value to set the random seed as.

    Returns
    -------
    RandomState
        An initialized numpy RandomState.

    Notes
    -----
    This sets the random state of a new RandomState object, independent of the global numpy state.
    """

    return np.random.RandomState(seed_value)
