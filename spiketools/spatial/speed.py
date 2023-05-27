"""Speed related functions."""

import numpy as np

from spiketools.spatial.distance import compute_distances
from spiketools.utils.timestamps import compute_sample_durations

###################################################################################################
###################################################################################################

def compute_speed(position, timestamps, align_output=True):
    """Compute speeds across a sequence of positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    timestamps : 1d array
        Timestamps, in seconds, corresponding to the position values.
    align_output : bool, optional, default: True
        If True, aligns the output with the sampling of the input, to match length.
        To do so, value of 0 is prepended to the output array.

    Returns
    -------
    speed : 1d array
        Vector of speeds across each position step.

    Examples
    --------
    Compute speed across a sequence of 1d positions:

    >>> position = np.array([1., 2., 4., 5.])
    >>> timestamps = np.array([0, 1, 2, 2.5])
    >>> compute_speed(position, timestamps)
    array([0., 1., 2., 2.])

    Compute speed across a sequence of 2d positions:

    >>> position = np.array([[1, 2, 2, 3],
    ...                      [1, 1, 2, 3]])
    >>> timestamps = np.array([0, 1, 1.5, 2.5])
    >>> compute_speed(position, timestamps)
    array([0.        , 1.        , 2.        , 1.41421356])
    """

    distances = compute_distances(position)
    durations = compute_sample_durations(timestamps, align_output=False)

    speed = distances / durations

    if align_output:
        speed = np.insert(speed, 0, 0)

    return speed
