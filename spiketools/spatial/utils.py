"""Spatial position related utility functions."""

import numpy as np

from spiketools.utils.data import compute_range
from spiketools.utils.checks import check_axis, check_array_orientation
from spiketools.spatial.checks import check_position, check_bin_definition, check_bin_widths

###################################################################################################
###################################################################################################

def get_position_xy(position, orientation=None):
    """Get the x & y data vectors from a 2d position data array.

    Parameters
    ----------
    position : 2d array
        Position values.
    orientation : {'row', 'column'}, optional
        The orientation of the position data.
        If not provided, is inferred from the given data.

    Returns
    -------
    x_data, y_data : 1d array
        Extracted X & Y position data.
    """

    assert position.ndim == 2, "Position data must be 2d to unpack X & Y dimensions."

    orientation = check_array_orientation(position, 2) if not orientation else orientation

    if orientation == 'row':
        x_data, y_data = position
    else:
        x_data, y_data = position[:, 0], position[:, 1]

    return x_data, y_data


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

    bins = check_bin_definition(bins)

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
    ranges : 1d array or list of 1d array
        Ranges for each dimension in the spatial data.

    Examples
    --------
    Compute the 2d position ranges for:
    (x, y) = (1.5, 6.5), (2.5, 7.5), (3.5, 8.5), (5.1, 9.1).

    >>> position = np.array([[1.5, 2.5, 3.5, 5.1], [6.5, 7.5, 8.5, 9.1]])
    >>> compute_pos_ranges(position)
    [array([1.5, 5.1]), array([6.5, 9.1])]

    Compute the 1d position range for:
    x = 1.5, 2.5, 3.5, 5.1

    >>> position = np.array([1.5, 2.5, 3.5, 5.1])
    >>> compute_pos_ranges(position)
    array([1.5, 5.1])
    """

    check_position(position)

    if position.ndim == 1:
        ranges = np.array(compute_range(position))

    elif position.ndim == 2:
        # Regardless of row / column input data, organizes output to have same orientation
        axis = check_axis(None, position)
        ranges = np.apply_along_axis(compute_range, axis, position)
        ranges = list(ranges.T if axis == 0 else ranges)

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

    Examples
    --------
    Compute bin width from an array of 5 bin edges:

    >>> bin_edges = [1.5, 3.5, 5.5, 7.5, 9.5]
    >>> compute_bin_width(bin_edges)
    2.0
    """

    bin_widths = np.diff(bin_edges)
    check_bin_widths(bin_widths)
    bin_width = bin_widths[0]

    return bin_width


def convert_2dindices(xbins, ybins, bins):
    """Convert a set of 2d bin indices into the equivalent 1d indices.

    Parameters
    ----------
    xbins, ybins : 1d array
        Bin assignment indices for the x and y dimension of a 2d binning.
    bins : list of [int, int]

    Returns
    -------
    indices : 1d array
        Bin assignment indices for a 1d binning.

    Notes
    -----
    The equivalent 1d indices for a set of 2d indices are those that access the
    equivalent values once the data array has been flattened into a 1d array.

    Examples
    --------
    Convert 2d bins with shape [3, 2] into the equivalent 1d indices:

    >>> xbins, ybins = np.array([2, 0, 1, 0, 1, 2]), np.array([0, 1, 1, 0, 0, 1])
    >>> bins = [3, 2]
    >>> convert_2dindices(xbins, ybins, bins)
    array([2, 3, 4, 0, 1, 5])
    """

    indices = np.ravel_multi_index((ybins, xbins), np.flip(bins))

    return indices


def convert_1dindices(indices, bins):
    """Convert a set of 1d bin indices into the equivalent 2d indices.

    Parameters
    ----------
    indices : 1d array
        Bin assignment indices for a 1d binning.
    bins : list of [int, int]
        The bin definition for dividing up the space.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].

    Returns
    -------
    xbins, ybins : 1d array
        Bin assignment indices for the x and y dimension of a 2d binning.

    Notes
    -----
    The equivalent 2d indices for a set of 1d indices are those that access the
    equivalent values once the data array has been reshaped to the given 2d bin definition.

    Examples
    --------
    Convert 1d bin indices into equivalent 2d indices with a [4, 4] shape:

    >>> indices = np.array([0, 4, 5, 6, 7, 3, 2, 1, 0])
    >>> bins = [3, 3]
    >>> convert_1dindices(indices, bins)
    (array([0, 1, 2, 0, 1, 0, 2, 1, 0]), array([0, 1, 1, 2, 2, 1, 0, 0, 0]))
    """

    ybins, xbins = np.unravel_index(indices, np.flip(bins))

    return xbins, ybins
