"""Spatial position and occupancy related functions."""

import warnings

import numpy as np
import pandas as pd

from spiketools.utils.data import assign_data_to_bins
from spiketools.utils.checks import check_array_orientation
from spiketools.utils.extract import create_mask, get_values_by_time_range
from spiketools.utils.timestamps import compute_sample_durations
from spiketools.spatial.checks import check_position, check_bin_definition
from spiketools.spatial.utils import get_position_xy

###################################################################################################
###################################################################################################

def compute_bin_edges(position, bins, area_range=None):
    """Compute spatial bin edges.

    Parameters
    ----------
    position : 1d or 2d array or None
        Position values. If None, area_range is required to define bins.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    area_range : list of list, optional
        Edges of the area to bin.
        For 1d defined as [min, max]. For 2d, defined as [[x_min, x_max], [y_min, y_max]].
        Any values outside this range will be considered outliers, and not used to compute edges.

    Returns
    -------
    x_edges : 1d array
        Edge definitions for the spatial binning.
    y_edges : 1d array
        Edge definitions for the spatial binning. Only returned in 2d case.

    Examples
    --------
    Compute bin edges for 1d position values:

    >>> position = np.array([1, 2, 3, 4, 5])
    >>> compute_bin_edges(position, bins=[5])
    array([1. , 1.8, 2.6, 3.4, 4.2, 5. ])

    Compute bin edges for 2d position values:

    >>> position = np.array([[1, 2, 3, 4, 5],
    ...                      [6, 7, 8, 9, 10]])
    >>> compute_bin_edges(position, bins=[5, 4])
    (array([1. , 1.8, 2.6, 3.4, 4.2, 5. ]), array([ 6.,  7.,  8.,  9., 10.]))
    """

    bins = check_bin_definition(bins, position)

    if len(bins) == 1:

        x_edges = np.histogram_bin_edges(position, bins=bins[0], range=area_range)

        return x_edges

    elif len(bins) == 2:

        x_pos, y_pos = get_position_xy(position) \
            if isinstance(position, np.ndarray) else (None, None)
        x_range, y_range = area_range if isinstance(area_range, list) else (None, None)

        x_edges = np.histogram_bin_edges(x_pos, bins=bins[0], range=x_range)
        y_edges = np.histogram_bin_edges(y_pos, bins=bins[1], range=y_range)

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
        Bin assignments for the y-dimension for each position. Only returned in 2d case.

    Notes
    -----
    - Values in the edge array(s) should be monotonically increasing.
    - If there are no outliers (all position values are within edge ranges), the returned
      bin assignments will range from (0, n_bins-1).
    - Outliers (position values beyond the given edges definitions), will be encoded as -1
      (left side) or `n_bins` (right side). If `check_range` is True, a warning will be raised.
    - By default position values equal to the left-most & right-most edges are treated as
      within the bounds (not treated as outliers), unless `include_edge` is set as False.

    Examples
    --------
    Compute bin assignment for 1d position values, given precomputed bin edges:

    >>> position = np.array([1.5, 2.5, 3.5, 5])
    >>> x_edges = np.array([1, 2, 3, 4, 5])
    >>> compute_bin_assignment(position, x_edges)
    array([0, 1, 2, 3])

    Compute bin assignment for 2d position values, given precomputed bin edges:

    >>> position = np.array([[1.5, 2.5, 3.5, 5],
    ...                      [6.5, 7.5, 8.5, 9]])
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

        x_pos, y_pos = get_position_xy(position)
        x_bins = assign_data_to_bins(x_pos, x_edges, check_range, include_edge)
        y_bins = assign_data_to_bins(y_pos, y_edges, check_range, include_edge)

        return x_bins, y_bins


def compute_bin_counts_pos(position, bins, area_range=None, occupancy=None, orientation=None):
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
    orientation : {'row', 'column'}, optional
        The orientation of the position data.
        If not provided, is inferred from the position data.

    Returns
    -------
    bin_counts : 1d or 2d array
        Amount of events in each bin.
        For 2d, has shape [n_y_bins, n_x_bins] (see notes).

    Notes
    -----
    For the 2d case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.

    Examples
    --------
    Compute counts across 2d bins from position data:

    >>> position = np.array([[0.5, 1.0, 1.5, 2.0, 3.0],
    ...                      [0.1, 0.2, 0.3, 0.4, 0.5]])
    >>> bins = [2, 3]
    >>> compute_bin_counts_pos(position, bins)
    array([[2, 0],
           [1, 0],
           [0, 2]])
    """

    bins = check_bin_definition(bins, position)

    if position.ndim == 1:
        bin_counts, _ = np.histogram(position, bins=bins[0], range=area_range)

    elif position.ndim == 2:
        bin_counts, _, _ = np.histogram2d(*get_position_xy(position, orientation=orientation),
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
    For the 2d case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.

    Examples
    --------
    Compute the bin counts per bin for 1d data, given precomputed x bin assignments:

    >>> bins = 3
    >>> xbins = [0, 2, 1, 0, 1]
    >>> compute_bin_counts_assgn(bins, xbins)
    array([2, 2, 1])

    Compute the bin counts for 1d data, given precomputed x & y bin assignments:

    >>> bins = [2, 2]
    >>> xbins = [0, 0, 0, 1]
    >>> ybins = [0, 0, 1, 1]
    >>> compute_bin_counts_assgn(bins, xbins, ybins)
    array([[2., 0.],
           [1., 1.]])
    """

    bins = check_bin_definition(bins)

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
    Normalized a pre-computed 2d bin counts array by occupancy:

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


def create_position_df(position, timestamps, bins, area_range=None, speed=None,
                       min_speed=None, max_speed=None, min_time=None, max_time=None,
                       dropna=True, check_range=True):
    """Create a dataframe that stores information about position bins.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    timestamps : 1d array
        Timestamps, in seconds, corresponding to the position values.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
    speed : 1d array, optional
        Current speed for each position.
        Should be the same length as timestamps.
    min_speed, max_speed : float, optional
        Minimum and/or maximum speed thresholds to apply.
        Any entries with an associated speed below the minimum or above maximum are dropped.
    min_time, max_time : float, optional
        Minimum and/or maximum time thresholds, per bin observation, to apply.
        Any entries with an associated time length below the minimum or above maximum are dropped.
    dropna : bool, optional, default: True
        If True, drops any rows from the dataframe that contain NaN values.
    check_range : bool, optional, default: True
        Whether to check the given bin definition range against the position values.

    Returns
    -------
    bindf : pd.DataFrame
        Dataframe representation of position bin information.
    """

    bins = check_bin_definition(bins, position)

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

    bindf = bindf[create_mask(bindf.time, min_time, max_time)]

    if speed is not None:
        bindf = bindf[create_mask(bindf.speed, min_speed, max_speed)]

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

    bins = check_bin_definition(bins)

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
                      min_speed=None, max_speed=None, min_time=None, max_time=None,
                      check_range=True, minimum=None, normalize=False, set_nan=False):
    """Compute occupancy across spatial bin positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    timestamps : 1d array
        Timestamps, in seconds, corresponding to the position values.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
    speed : 1d array, optional
        Current speed for each position.
        Should be the same length as timestamps.
    min_speed, max_speed : float, optional
        Minimum and/or maximum speed thresholds to apply.
        Any entries with an associated speed below the minimum or above maximum are dropped.
    min_time, max_time : float, optional
        Minimum and/or maximum time thresholds, per bin observation, to apply.
        Any entries with an associated time length below the minimum or above maximum are dropped.
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
    For the 2d case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.

    Examples
    --------
    Compute occupancy for a set of 1d position values:

    >>> position = np.array([1.0, 1.5, 2.5, 3.5, 5])
    >>> timestamps = np.linspace(0, 1, position.shape[0])
    >>> compute_occupancy(position, timestamps, bins=[4])
    array([0.5 , 0.25, 0.25, 0.  ])

    Compute occupancy for a set of 2d position values:

    >>> position = np.array([[1.0, 2.5, 1.5, 3.0, 3.5, 5.0],
    ...                      [5.0, 7.5, 6.5, 5.0, 8.5, 9.0]])
    >>> timestamps = np.linspace(0, 1, position.shape[1])
    >>> compute_occupancy(position, timestamps, bins=[2, 2])
    array([[0.4, 0.2],
           [0.2, 0.2]])
    """

    df = create_position_df(position, timestamps, bins, area_range,
                            speed, min_speed, max_speed, min_time, max_time,
                            check_range=check_range)
    occupancy = compute_occupancy_df(df, bins, minimum, normalize, set_nan)

    return occupancy


def compute_trial_occupancy(position, timestamps, bins, start_times, stop_times,
                            area_range=None, speed=None, min_speed=None, max_speed=None,
                            min_time=None, max_time=None, orientation=None, **occupancy_kwargs):
    """Compute trial-level occupancy across spatial bin positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    timestamps : 1d array
        Timestamps, in seconds, corresponding to the position values.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    start_times, stop_times : 1d array
        The start and stop times, in seconds, of each trial.
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
    speed : 1d array, optional
        Current speed for each position.
        Should be the same length as timestamps.
    min_speed, max_speed : float, optional
        Minimum and/or maximum speed thresholds to apply.
        Any entries with an associated speed below the minimum or above maximum are dropped.
    min_time, max_time : float, optional
        Minimum and/or maximum time thresholds, per bin observation, to apply.
        Any entries with an associated time length below the minimum or above maximum are dropped.
    orientation : {'row', 'column'}, optional
        The orientation of the position data.
        If not provided, is inferred from the position data.
    occupancy_kwargs
        Additional arguments to pass into the the `compute_occupancy` function.

    Returns
    -------
    trial_occupancy : ndarray
        Occupancy data across trials.

    Examples
    --------
    Compute trial-level occupancy for 1d position data:

    >>> bins = 2
    >>> position = np.array([1, 3, 5, 6, 9, 10, 2, 4, 6, 7, 9])
    >>> timestamps = np.linspace(0, 50, len(position))
    >>> start_times, stop_times = [0, 25], [26, 50]
    >>> compute_trial_occupancy(position, timestamps, bins, start_times, stop_times)
    array([[15., 10.],
           [10., 15.]])

    Compute trial-level occupancy for 2d position data:

    >>> bins = [2, 3]
    >>> position = np.array([[1, 2, 4, 4.5, 5, 2, 2.5, 3.5, 4, 5, 5.5],
    ...                      [6, 7, 8, 8.5, 9.5, 10, 6, 6.5, 7, 8, 10]])
    >>> timestamps = np.linspace(0, 50, position.shape[1])
    >>> start_times, stop_times = [0, 25], [26, 50]
    >>> compute_trial_occupancy(position, timestamps, bins, start_times, stop_times)
    array([[[10.,  0.],
            [ 0., 10.],
            [ 0.,  5.]],
    <BLANKLINE>
           [[10.,  5.],
            [ 0.,  5.],
            [ 5.,  0.]]])
    """

    bins = check_bin_definition(bins, position)
    orientation = check_array_orientation(position, len(bins)) if not orientation else orientation

    t_speed = None
    trial_occupancy = np.zeros([len(start_times), *np.flip(bins)])
    for ind, (start, stop) in enumerate(zip(start_times, stop_times)):

        t_times, t_pos = get_values_by_time_range(timestamps, position, start, stop)

        if speed is not None:
            _, t_speed = get_values_by_time_range(timestamps, speed, start, stop)

        trial_occupancy[ind, :] = compute_occupancy(\
            t_pos, t_times, bins, area_range, t_speed,
            min_speed, max_speed, min_time, max_time,
            **occupancy_kwargs)

    return trial_occupancy
