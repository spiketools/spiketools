"""Spatial position and occupancy related functions."""

import warnings

import numpy as np
import pandas as pd

from spiketools.utils.data import assign_data_to_bins
from spiketools.spatial.checks import check_position, check_position_bins
from spiketools.spatial.utils import compute_sample_durations

###################################################################################################
###################################################################################################

def compute_bin_edges(position, bins, area_range=None):
    """Compute spatial bin edges.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
        Any values outside this range will be considered outliers, and not used to compute edges.

    Returns
    -------
    x_edges : 1d array
        Edge definitions for the spatial binning.
    y_edges : 1d array
        Edge definitions for the spatial binning. Only returned in 2D case.

    Examples
    --------
    Compute bin edges for 1D position values:

    >>> position = np.array([1, 2, 3, 4, 5])
    >>> compute_bin_edges(position, bins=[5])
    array([1. , 1.8, 2.6, 3.4, 4.2, 5. ])

    Compute bin edges for 2D position values:

    >>> position = np.array([[1, 2, 3, 4, 5], \
                             [6, 7, 8, 9, 10]])
    >>> compute_bin_edges(position, bins=[5, 4])
    (array([1. , 1.8, 2.6, 3.4, 4.2, 5. ]), array([ 6.,  7.,  8.,  9., 10.]))
    """

    bins = check_position_bins(bins, position)

    if position.ndim == 1:
        _, x_edges = np.histogram(position, bins=bins[0], range=area_range)

        return x_edges

    elif position.ndim == 2:
        _, x_edges, y_edges = np.histogram2d(position[0, :], position[1, :],
                                             bins=bins, range=area_range)
        return x_edges, y_edges


def compute_bin_assignment(position, x_edges, y_edges=None, check_range=True, include_edge=True):
    """Compute spatial bin assignment.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    x_edges : 1d array
        Edge definitions for the x dimension of the spatial binning.
    y_edges : 1d array, optional, default: None
        Edge definitions for the y dimension of the spatial binning. Only used if position is 2d.
    check_range : bool, optional, default: True
        Whether to check if the given edges fully cover the given data.
        If True, runs a check that raises a warning if any data values exceed edge ranges.
    include_edge : bool, optional, default: True
        Whether to include positions on the edge into the bin.

    Returns
    -------
    x_bins : 1d array
        Bin assignments for the x-dimension for each position.
    y_bins : 1d array
        Bin assignments for the y-dimension for each position. Only returned in 2D case.

    Notes
    -----
    - Values in the edge array(s) should be monotonically increasing.
    - If there are no outliers (all position values are between edge ranges), the returned
      bin assignments will range from (0, n_bins-1).
    - Outliers (position values beyond the given edges definitions), will be encoded as -1
      (left side) or `n_bins` (right side). If `check_range` is True, a warning will be raised.
    - By default position values equal to the left-most & right-most edges are treated as
      within the bounds (not treated as outliers), unless `include_edge` is set as False.

    Examples
    --------
    Compute bin assignment for 1D position values, given precomputed bin edges:

    >>> position = np.array([1.5, 2.5, 3.5, 5])
    >>> x_edges = np.array([1, 2, 3, 4, 5])
    >>> compute_bin_assignment(position, x_edges)
    array([0, 1, 2, 3])

    Compute bin assignment for 2D position values, given precomputed bin edges:

    >>> position = np.array([[1.5, 2.5, 3.5, 5], \
                             [6.5, 7.5, 8.5, 9]])
    >>> x_edges = np.array([1, 2, 3, 4, 5])
    >>> y_edges = np.array([6, 7, 8, 9, 10])
    >>> compute_bin_assignment(position, x_edges, y_edges)
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    """

    check_position(position)

    if position.ndim == 1:

        x_bins = assign_data_to_bins(position, x_edges, check_range, include_edge)

        return x_bins

    elif position.ndim == 2:

        x_bins = assign_data_to_bins(position[0, :], x_edges, check_range, include_edge)
        y_bins = assign_data_to_bins(position[1, :], y_edges, check_range, include_edge)

        return x_bins, y_bins


def compute_bin_counts_pos(position, bins, area_range=None, occupancy=None):
    """Compute counts per bin, from position data.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
        Any values outside this range will be considered outliers, and not used to compute edges.
    occupancy : 1d or 2d array, optional
        Occupancy across the spatial bins.
        If provided, used to normalize bin counts.

    Returns
    -------
    bin_counts : 1d or 2d array
        Amount of events in each bin.
        For 2d, has shape [n_y_bins, n_x_bins] (see notes).

    Notes
    -----
    For the 2D case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.

    Examples
    --------
    Compute counts across 2d bins from position data:

    >>> position = np.array([[0.5, 1.0, 1.5, 2.0, 3.0], \
                             [0.1, 0.2, 0.3, 0.4, 0.5]])
    >>> bins = [2, 3]
    >>> compute_bin_counts_pos(position, bins)
    array([[2, 0],
           [1, 0],
           [0, 2]])
    """

    bins = check_position_bins(bins, position)

    if position.ndim == 1:
        bin_counts, _ = np.histogram(position, bins=bins[0], range=area_range)

    elif position.ndim == 2:
        bin_counts, _, _ = np.histogram2d(position[0, :], position[1, :],
                                          bins=bins, range=area_range)
        bin_counts = bin_counts.T

    bin_counts = bin_counts.astype('int')
    if occupancy is not None:
        bin_counts = normalize_bin_counts(bin_counts, occupancy)

    return bin_counts


def compute_bin_counts_assgn(bins, xbins, ybins=None, occupancy=None):
    """Compute number of counts per bin, from bin assignments.

    Parameters
    ----------
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    xbins : 1d array
        Bin assignments for the x-dimension.
    ybins : 1d array, optional
        Bin assignments for the y-dimension.
    occupancy : 1d or 2d array, optional
        Occupancy across the spatial bins.
        If provided, used to normalize bin counts.

    Returns
    -------
    bin_counts : 1d or 2d array
        Amount of counts in each bin.
        For 2d, has shape [n_y_bins, n_x_bins] (see notes).

    Notes
    -----
    For the 2D case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.

    Examples
    --------
    Compute the bin counts per bin for 1D data, given precomputed x bin assignments:

    >>> bins = 3
    >>> xbins = [0, 2, 1, 0, 1]
    >>> compute_bin_counts_assgn(bins, xbins)
    array([2, 2, 1])

    Compute the bin counts for 2D data, given precomputed x & y bin assignments:

    >>> bins = [2, 2]
    >>> xbins = [0, 0, 0, 1]
    >>> ybins = [0, 0, 1, 1]
    >>> compute_bin_counts_assgn(bins, xbins, ybins)
    array([[2., 0.],
           [1., 1.]])
    """

    bins = check_position_bins(bins)

    if ybins is None:
        bins = np.arange(0, bins[0] + 1)
        bin_counts, _ = np.histogram(xbins, bins=bins)
    else:
        bins = [np.arange(0, bins[0] + 1), np.arange(0, bins[1] + 1)]
        bin_counts, _, _ = np.histogram2d(xbins, ybins, bins=bins)
        bin_counts = bin_counts.T

    if occupancy is not None:
        bin_counts = normalize_bin_counts(bin_counts, occupancy)

    return bin_counts


def normalize_bin_counts(bin_counts, occupancy):
    """Normalize bin counts by occupancy.

    Parameters
    ----------
    bin_counts : 1d or 2d array
        Bin counts.
    occupancy : 1d or 2d array
        Spatially binned occupancy.

    Returns
    -------
    normalized_bin_counts : 1d or 2d array
        Normalized bin counts.

    Notes
    -----
    For any bins in which the occupancy is zero, the output will NaN.

    Examples
    --------
    Normalized a pre-computed 2D bin counts array by occupancy:

    >>> bin_counts = np.array([[0, 1, 0], [1, 2, 0]])
    >>> occupancy = np.array([[0, 2, 1], [1, 1, 0]])
    >>> normalize_bin_counts(bin_counts, occupancy)
    array([[nan, 0.5, 0. ],
           [1. , 2. , nan]])
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        normalized_bin_counts = bin_counts / occupancy

    return normalized_bin_counts


def create_position_df(position, timestamps, bins, area_range=None,
                       speed=None, speed_threshold=None, time_threshold=None,
                       dropna=True, check_range=True):
    """Create a dataframe that stores information about position bins.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    timestamps : 1d array
        Timestamps.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
    speed : 1d array, optional
        Current speed for each position.
        Should be the same length as timestamps.
    speed_threshold : float, optional
        A minimum speed threshold to apply.
        If provided, any position values with an associated speed below this value are dropped.
    time_threshold : float, optional
        A maximum time threshold, per bin observation, to apply.
        If provided, any bin values with an associated time length above this value are dropped.
    dropna : bool, optional, default: True
        If True, drops any rows from the dataframe that contain NaN values.
    check_range : bool, optional, default: True
        Whether to check the given bin definition range against the position values.

    Returns
    -------
    bindf : pd.DataFrame
        Dataframe representation of position bin information.
    """

    bins = check_position_bins(bins, position)

    data_dict = {'time' : compute_sample_durations(timestamps)}
    if speed is not None:
        data_dict['speed'] = speed

    if position.ndim == 1:

        # Spatially bin 1d position data, and collect bin assignment information
        x_edges = compute_bin_edges(position, bins, area_range)
        x_bins = compute_bin_assignment(position, x_edges, check_range=check_range)

        data_dict['xbin'] = pd.Categorical(\
            x_bins, categories=list(range(0, bins[0])), ordered=True)

    elif position.ndim == 2:

        # Spatially bin 2d position data, and collect bin assignment information
        x_edges, y_edges = compute_bin_edges(position, bins, area_range)
        x_bins, y_bins = compute_bin_assignment(position, x_edges, y_edges,
                                                check_range=check_range)

        data_dict['xbin'] = pd.Categorical(\
            x_bins, categories=list(range(0, bins[0])), ordered=True)
        data_dict['ybin'] = pd.Categorical(\
            y_bins, categories=list(range(0, bins[1])), ordered=True)

    bindf = pd.DataFrame(data_dict)

    if time_threshold is not None:
        bindf = bindf[bindf.time < time_threshold]

    if speed_threshold is not None:
        bindf = bindf[bindf.speed > speed_threshold]

    if dropna:
        bindf = bindf.dropna()

    return bindf


def compute_occupancy_df(bindf, bins, minimum=None, normalize=False, set_nan=False):
    """Compute the bin occupancy from bin-position dataframe.

    Parameters
    ----------
    bindf : pd.DataFrame
        Dataframe representation of position bin information.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    minimum : float, optional
        The minimum required occupancy.
        If defined, any values below this are set to zero.
    normalize : bool, optional, default: False
        Whether to normalize occupancy to sum to 1.
    set_nan : bool, optional, default: False
        Whether to set zero occupancy locations as NaN.

    Returns
    -------
    occupancy : 1d or 2d array
        Computed occupancy across the space.
        For 2d, has shape [n_y_bins, n_x_bins] (see notes in `compute_occupancy`).
    """

    bins = check_position_bins(bins)

    # Group position samples into spatial bins, summing total time spent there
    groupby = sorted([el for el in list(bindf.columns) if 'bin' in el])
    bingroup = bindf.groupby(groupby)['time'].sum()
    # This standardizes any unobserved values to be zero
    #   This can otherwise vary by code version (sometimes being NaN instead)
    bingroup = bingroup.fillna(0)

    # Extract and re-organize occupancy into array
    occupancy = bingroup.values.reshape(*bins)
    occupancy = occupancy.T

    if minimum:
        occupancy[occupancy < minimum] = 0.

    if normalize:
        occupancy = occupancy / np.sum(occupancy)

    if set_nan:
        occupancy[occupancy == 0.] = np.nan

    return occupancy


def compute_occupancy(position, timestamps, bins, area_range=None, speed=None,
                      speed_threshold=None, time_threshold=None, check_range=True,
                      minimum=None, normalize=False, set_nan=False):
    """Compute occupancy across spatial bin positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    timestamps : 1d array
        Timestamps.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
    speed : 1d array, optional
        Current speed for each position.
        Should be the same length as timestamps.
    speed_threshold : float, optional
        Speed threshold to apply.
        If provided, any position values with an associated speed below this value are dropped.
    time_threshold : float, optional
        A maximum time threshold, per bin observation, to apply.
        If provided, any bin values with an associated time length above this value are dropped.
    check_range : bool, optional, default: True
        Whether to check the given bin definition range against the position values.
    minimum : float, optional
        The minimum required occupancy.
        If defined, any values below this are set to zero.
    normalize : bool, optional, default: False
        Whether to normalize occupancy to sum to 1.
    set_nan : bool, optional, default: False
        Whether to set zero occupancy locations as NaN.

    Returns
    -------
    occupancy : 1d or 2d array
        Computed occupancy across the space.
        For 2d, has shape [n_y_bins, n_x_bins] (see notes).

    Notes
    -----
    For the 2D case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.

    Examples
    --------
    Compute occupancy for a set of 1D position values:

    >>> position = np.array([1.0, 1.5, 2.5, 3.5, 5])
    >>> timestamps = np.linspace(0, 1, position.shape[0])
    >>> compute_occupancy(position, timestamps, bins=[4])
    array([0.5 , 0.25, 0.25, 0.  ])

    Compute occupancy for a set of 2D position values:

    >>> position = np.array([[1.0, 2.5, 1.5, 3.0, 3.5, 5.0], \
                             [5.0, 7.5, 6.5, 5.0, 8.5, 9.0]])
    >>> timestamps = np.linspace(0, 1, position.shape[1])
    >>> compute_occupancy(position, timestamps, bins=[2, 2])
    array([[0.4, 0.2],
           [0.2, 0.2]])
    """

    df = create_position_df(position, timestamps, bins, area_range,
                            speed, speed_threshold, time_threshold,
                            check_range=check_range)
    occupancy = compute_occupancy_df(df, bins, minimum, normalize, set_nan)

    return occupancy
