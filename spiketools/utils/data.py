"""Utility functions for working with data arrays."""

from copy import deepcopy

import numpy as np

from scipy.ndimage import gaussian_filter

from spiketools.utils.checks import check_param_options, check_bin_range

###################################################################################################
###################################################################################################

def compute_range(data):
    """Compute the range of an array of data.

    Parameters
    ----------
    data : array
        Array of numerical data.

    Returns
    -------
    min, max : float
        Minimum and maximum values of the data array.
    """

    return np.nanmin(data), np.nanmax(data)


def smooth_data(data, sigma):
    """Smooth an array of data, using a gaussian kernel.

    Parameters
    ----------
    data : ndarray
        Data to smooth.
    sigma : float
        Standard deviation of the gaussian kernel to apply for smoothing.

    Returns
    -------
    data : 2d array
        The smoothed data.

    Notes
    -----
    This function is applied on a copy of the data (to not change the original).
    Any NaN values will be set as 0 for smoothing purposes.
    """

    data = deepcopy(data)
    data[np.isnan(data)] = 0

    data = gaussian_filter(data, sigma=sigma)

    return data


def drop_nans(data):
    """Drop any NaNs values from an array.

    Parameters
    ----------
    data : 1d or 2d array
        Data array to check and drop NaNs from.

    Returns
    -------
    data : 1d or 2d array
        Data array with NaNs removed.

    Notes
    -----
    For 2d arrays, this function assumes the same columns to be NaN across all rows.
    """

    nans = np.isnan(data)

    if data.ndim == 1:
        data = data[np.where(~nans)]
    elif data.ndim == 2:
        data = data[~nans].reshape(nans.shape[0], sum(~nans[0, :]))
    else:
        raise ValueError('Only 1d or 2d arrays supported.')

    return data


def assign_data_to_bins(data, edges, include_edge=True):
    """Assign data values to data bins, based on given edges.

    Parameters
    ----------
    data : 1d array
        Data values to bin.
    edges : 1d array
        Edge definitions for the binning.
    include_edge : bool, optional, default: True
        Whether to include data values on the edge into the bin.

    Returns
    -------
    assignments : 1d array
        Bin assignments per data value.
    """

    check_bin_range(data, edges)
    assignments = np.digitize(data, edges, right=False)

    if include_edge:
        assignments = _include_bin_edge(assignments, data, edges, side='left')

    return assignments - 1


def _include_bin_edge(assignments, position, edges, side='left'):
    """Update bin assignment so last bin includes edge values.

    Parameters
    ----------
    assignments : 1d array
        The bin assignment for each position.
    position : 1d array
        Position values.
    edges : 1d array
        The bin edges.
    side : {'left', 'right'}
        Which side was used to compute bin assignment.

    Returns
    -------
    assignments : 1d array
        The bin assignment for each position.

    Notes
    -----
    For any position values that exactly match the left-most or right-most bin edges, by default
    (from np.digitize), one of these sides will be considered an outlier. This is because bin
    assignment is computed as `pos >= left_bin_edge & pos < right_bin_edge (flipped if right=True).
    To address this, this function resets position values == edges as with the bin on the edge.
    """

    check_param_options(side, 'side', ['left', 'right'])

    if side == 'left':

        # If side left, right position == edge gets set as len(bins), so decrement by 1
        mask = position == edges[-1]
        assignments[mask] = assignments[mask] - 1

    elif side == 'right':

        # If side right, left position == edge gets set as 0, so increment by 1
        mask = position == edges[0]
        assignments[mask] = assignments[mask] + 1

    return assignments
