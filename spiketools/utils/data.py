"""Utility functions for working with data arrays."""

from copy import deepcopy

import numpy as np

from scipy.ndimage import gaussian_filter

from spiketools.utils.checks import check_array_orientation, check_param_options, check_bin_range

###################################################################################################
###################################################################################################

def make_orientation(arr, to_orientation, from_orientation=None):
    """Check and make sure a 2d array is in a specified orientation.

    Parameters
    ----------
    arr : 2d array
        Array to check orientation.
    to_orientation : {'row', 'column'}
        The desired orientation of the output data.
        If the input is not already in this orientation, is transposed.
    from_orientation : {'row', 'column'}, optional
        The orientation of the input array.
        If not provided, is inferred from the input data.

    Returns
    -------
    arr : 2d array
        2d array in specified orientation.
    """

    if not from_orientation:
        from_orientation = check_array_orientation(arr)

    if to_orientation != from_orientation:
        arr = arr.T

    return arr


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

    Examples
    --------
    Compute the range of some position data:

    >>> data = np.array([1.5, 1, 0.5, 2, 3, 2.5])
    >>> compute_range(data)
    (0.5, 3.0)
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

    Examples
    --------
    Smooth a 1d data array using a gaussian kernel:

    >>> data = np.array([1, 3, 5, 7, 9])
    >>> smooth_data(data, sigma=0.8)
    array([1, 3, 5, 6, 8])
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

    Examples
    --------
    Drop all NaNs values from a 1d array:

    >>> data = np.array([1, 2, 3.5, np.nan, 6, 2, np.nan, 1])
    >>> drop_nans(data)
    array([1. , 2. , 3.5, 6. , 2. , 1. ])
    """

    nans = np.isnan(data)

    if data.ndim == 1:
        data = data[np.where(~nans)]
    elif data.ndim == 2:
        data = data[~nans].reshape(nans.shape[0], sum(~nans[0, :]))
    else:
        raise ValueError('Only 1d or 2d arrays supported.')

    return data


def permute_vector(data, n_permutations=1000):
    """Create permutations of a vector of data.

    Parameters
    ----------
    data : 1d array
        Vector to permute.
    n_permutations : int, optional, default: 1000
        Number of permutations to do.

    Returns
    -------
    permutations : 2d array
        Permutations of the input data.

    Notes
    -----
    Code adapted from here: https://stackoverflow.com/questions/46859304/

    This function doesn't have any randomness - for a given array it will
    iterate through the same set of permutations, in sequence.

    Examples
    --------
    Create permutations for a vector of data:

    >>> data = np.array([0, 5, 10, 15, 20])
    >>> permute_vector(data, n_permutations=4)
    array([[ 0,  5, 10, 15, 20],
           [ 5, 10, 15, 20,  0],
           [10, 15, 20,  0,  5],
           [15, 20,  0,  5, 10]])
    """

    assert data.ndim == 1, 'The permute_vector function only works on 1d arrays.'

    data_ext = np.concatenate((data, data[:-1]))
    strides = data.strides[0]
    permutations = np.lib.stride_tricks.as_strided(data_ext,
                                                   shape=(n_permutations, len(data)),
                                                   strides=(strides, strides),
                                                   writeable=False).copy()

    return permutations


def assign_data_to_bins(data, edges, check_range=True, include_edge=True):
    """Assign data values to data bins, based on given edges.

    Parameters
    ----------
    data : 1d array
        Data values to bin.
    edges : 1d array
        Edge definitions for the binning.
    check_range : bool, optional, default: True
        Whether to check if the given edges fully cover the given data.
        If True, runs a check that raises a warning if any data values exceed edge ranges.
    include_edge : bool, optional, default: True
        Whether to include data values on the edge into the bin.

    Returns
    -------
    assignments : 1d array
        Bin assignments per data value.

    Examples
    --------
    Assign data values into bins given the bin edges:

    >>> data = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4])
    >>> edges = np.array([0, 0.5, 1, 1.5])
    >>> assign_data_to_bins(data, edges)
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    """

    if check_range:
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

    Examples
    --------
    Update bin assignment of some position data using left side bin edges:

    >>> position = np.array([0.5, 1, 1.5, 2, 1.5, 3])
    >>> assignments = np.array([0, 1, 1, 2, 1, 3])
    >>> edges = np.array([0, 1, 2, 3])
    >>> _include_bin_edge(assignments, position, edges, side='left')
    array([0, 1, 1, 2, 1, 2])
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
