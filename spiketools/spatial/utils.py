"""Spatial position related utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def get_pos_ranges(positions):
    """Compute the range of positions.

    Parameters
    ----------
    positions : 2d array
        Position data.

    Returns
    -------
    ranges : list of list of float
        Ranges for each dimension in the spatial data.
    """

    ranges = []
    for dim in range(positions.shape[0]):
        ranges.append([np.min(positions[dim, :]), np.max(positions[dim, :])])

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
