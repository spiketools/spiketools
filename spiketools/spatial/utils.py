"""Spatial position related utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def get_pos_ranges(positions):
    """Compute the range of positions.

    Parameters
    ----------
    positions : 1d or 2d array
        Position data.

    Returns
    -------
    ranges : list of list of float
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
    [[1.5, 5.0]]
    """

    ranges = []

    # 2+d case
    if (len(positions.shape) > 1):
        for dim in range(positions.shape[0]):
            ranges.append([np.min(positions[dim, :]), np.max(positions[dim, :])])

    # 1d case
    else:
        ranges.append([np.min(positions[:]), np.max(positions[:])])

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
