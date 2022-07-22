"""Spatial position related utility functions."""

import numpy as np

from spiketools.utils.data import compute_range
from spiketools.spatial.checks import check_position_bins

###################################################################################################
###################################################################################################

def compute_nbins(bins):
    """Compute the number of bins for a given bin definition.

    Parameters
    ----------
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].

    Returns
    -------
    n_bins : int
        The total number of bins for the given bin definition.

    Examples
    --------
    Compute the number of bins for a given bin definition:

    >>> compute_nbins(bins=[4, 5])
    20
    """

    bins = check_position_bins(bins)

    if len(bins) == 1:
        n_bins = bins[0]
    else:
        n_bins = bins[0] * bins[1]

    return n_bins


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


def compute_bin_time(timestamps):
    """Compute the time duration of each position sample.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.

    Returns
    -------
    1d array
        Width, in time, of each bin.

    Examples
    --------
    Compute times between timestamp samples:

    >>> timestamp = np.array([0, 1.0, 3.0, 6.0, 8.0, 9.0])
    >>> compute_bin_time(timestamp)
    array([1., 2., 3., 2., 1., 0.])
    """

    return np.append(np.diff(timestamps), 0)


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
    bins : list of [int, int]
        The bin definition for dividing up the space.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].

    Returns
    -------
    indices : 1d array
        Bin assignment indices for a 1D binning.

    Notes
    -----
    The equivalent 1D indices for a set of 2D indices are those that access the
    equivalent values once the data array has been flattened into a 1d array.
    """

    indices = np.ravel_multi_index((ybins, xbins), np.flip(bins))

    return indices


def convert_1dindices(indices, bins):
    """Convert a set of 1D bin indices into the equivalent 2D indices.

    Parameters
    ----------
    indices : 1d array
        Bin assignment indices for a 1D binning.
    bins : list of [int, int]
        The bin definition for dividing up the space.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].

    Returns
    -------
    xbins, ybins : 1d array
        Bin assignment indices for the x and y dimension of a 2D binning.

    Notes
    -----
    The equivalent 2D indices for a set of 1d indices are those that access the
    equivalent values once the data array has been reshaped to the given 2D bin definition.
    """

    ybins, xbins = np.unravel_index(indices, np.flip(bins))

    return xbins, ybins
