"""Spatial position related utility functions."""

import numpy as np

from spiketools.utils.data import get_range

###################################################################################################
###################################################################################################

def get_pos_ranges(position):
    """Compute the range of positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position data.

    Returns
    -------
    ranges : list of float or list of list of float
        Ranges for each dimension in the spatial data.

    Examples
    --------
    Get 2D position ranges for:
    (x, y) = (1.5, 6.5), (2.5, 7.5), (3.5, 8.5), (5, 9).

    >>> position = np.array([[1.5, 2.5, 3.5, 5], [6.5, 7.5, 8.5, 9]])
    >>> get_pos_ranges(position)
    [[1.5, 5.0], [6.5, 9.0]]

    Get 1D position ranges for:
    x = 1.5, 2.5, 3.5, 5.

    >>> position = np.array([1.5, 2.5, 3.5, 5])
    >>> get_pos_ranges(position)
    [1.5, 5.0]
    """

    if position.ndim == 1:
        ranges = [*get_range(position)]

    elif position.ndim == 2:
        ranges = []
        for dim in range(position.shape[0]):
            ranges.append([*get_range(position[dim, :])])

    else:
        raise ValueError('Position input should be 1d or 2d.')

    return ranges


def get_bin_width(bins):
    """Compute bin width from a set of bin edges.

    Parameters
    ----------
    bins : 1d array
        Bin edges.

    Returns
    -------
    float
        The bin width.
    """

    return np.diff(bins)[0]
