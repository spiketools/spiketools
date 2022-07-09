"""Spatial position related utility functions."""

import numpy as np

from spiketools.utils.data import compute_range

###################################################################################################
###################################################################################################

def compute_pos_ranges(position):
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
    Compute the 2D position ranges for:
    (x, y) = (1.5, 6.5), (2.5, 7.5), (3.5, 8.5), (5, 9).

    >>> position = np.array([[1.5, 2.5, 3.5, 5], [6.5, 7.5, 8.5, 9]])
    >>> compute_pos_ranges(position)
    [[1.5, 5.0], [6.5, 9.0]]

    Compute the 1D position range for:
    x = 1.5, 2.5, 3.5, 5.

    >>> position = np.array([1.5, 2.5, 3.5, 5])
    >>> compute_pos_ranges(position)
    [1.5, 5.0]
    """

    if position.ndim == 1:
        ranges = [*compute_range(position)]

    elif position.ndim == 2:
        ranges = []
        for dim in range(position.shape[0]):
            ranges.append([*compute_range(position[dim, :])])

    else:
        raise ValueError('Position input should be 1d or 2d.')

    return ranges


def compute_bin_width(bin_edges):
    """Compute bin width from a set of bin edges.

    Parameters
    ----------
    bin_edges : 1d array
        Bin edges.

    Returns
    -------
    float
        The bin width.
    """

    return np.diff(bin_edges)[0]


def convert_2dindices(xbins, ybins, bins):
    """Convert a set of 2D bin indices into the equivalent 1D indices.

    Parameters
    ----------
    xbins, ybins : 1d array
        Bin assignment indices for the x and y dimension of a 2D binning.
    bins : list of int
        Bin definition.

    Returns
    -------
    indices : 1d array
        Bin assignment indices for a 1D binning.

    Notes
    -----
    The equivalent 1D indices for a set of 2D indices are those that access the
    equivalent values once the data array has been flattened into a 1d array.
    """

    indices = np.ravel_multi_index((xbins, ybins), bins)

    return indices


def convert_1dindices(indices, bins):
    """Convert a set of 1D bin indices into the equivalent 2D indices.

    Parameters
    ----------
    indices : 1d array
        Bin assignment indices for a 1D binning.
    bins : list of int
        Bin definition.

    Returns
    -------
    xbins, ybins : 1d array
        Bin assignment indices for the x and y dimension of a 2D binning.

    Notes
    -----
    The equivalent 2D indices for a set of 1d indices are those that access the
    equivalent values once the data array has been reshaped to the given 2D bin definition.
    """

    xbins, ybins = np.unravel_index(indices, bins)

    return xbins, ybins
