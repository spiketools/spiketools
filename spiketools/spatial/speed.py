"""Speed related functions."""

import numpy as np

from spiketools.spatial.distance import compute_distances

###################################################################################################
###################################################################################################

def compute_speed(position, bin_times):
    """Compute speeds across a sequence of positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    bin_times : 1d array
        Times spent traversing each position bin.

    Returns
    -------
    speed : 1d array
        Vector of speeds across each position step.

    Examples
    --------
    Compute speed across a sequence of 1d positions:

    >>> position = np.array([1., 2., 4., 5.])
    >>> bin_times = np.array([1, 1, 0.5])
    >>> compute_speed(position, bin_times)
    array([1., 2., 2.])

    Compute speed across a sequence of 2d positions:

    >>> position = np.array([[1, 2, 2, 3],
    ...                      [1, 1, 2, 3]])
    >>> bin_times = np.array([1, 0.5, 1])
    >>> compute_speed(position, bin_times)
    array([1.        , 2.        , 1.41421356])
    """

    distances = compute_distances(position)
    speed = distances / bin_times

    return speed
