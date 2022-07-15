"""Spatial position and occupancy related functions."""

import warnings

import numpy as np
import pandas as pd

from spiketools.utils.checks import check_param_options, check_bin_range
from spiketools.spatial.checks import check_position, check_position_bins
from spiketools.spatial.utils import compute_bin_time

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

    Compute bin edges for 2D position values, with x-range of 1-5 & y-range of 6-10:

    >>> position = np.array([[1, 2, 3, 4, 5],
    ...                      [6, 7, 8, 9, 10]])
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


def compute_bin_assignment(position, x_edges, y_edges=None, include_edge=True):
    """Compute spatial bin assignment.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    x_edges : 1d array
        Edge definitions for the spatial binning.
        Values within the arrays should be monotonically increasing.
    y_edges : 1d array, optional, default: None
        Edge definitions for the spatial binning.
        Values within the arrays should be monotonically increasing.
        Used in 2d case only.
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
    - In the case of zero outliers (all positions are between edge ranges), the returned
      values are encoded as bin position, with values between {0, n_bins-1}.
    - If there are outliers (some position values that are outside the given edges definitions),
      these are encoded as -1 (left side) or n_bins (right side). A warning will be raised.
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

    >>> position = np.array([[1.5, 2.5, 3.5, 5],
    ...                      [6.5, 7.5, 8.5, 9]])
    >>> x_edges = np.array([1, 2, 3, 4, 5])
    >>> y_edges = np.array([6, 7, 8, 9, 10])
    >>> compute_bin_assignment(position, x_edges, y_edges)
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    """

    check_position(position)

    if position.ndim == 1:

        check_bin_range(position, x_edges)
        x_bins = np.digitize(position, x_edges, right=False)

        if include_edge:
            x_bins = _include_bin_edge(position, x_bins, x_edges, side='left')

        return x_bins - 1

    elif position.ndim == 2:

        check_bin_range(position[0, :], x_edges)
        check_bin_range(position[1, :], y_edges)
        x_bins = np.digitize(position[0, :], x_edges, right=False)
        y_bins = np.digitize(position[1, :], y_edges, right=False)

        if include_edge:
            x_bins = _include_bin_edge(position[0, :], x_bins, x_edges, side='left')
            y_bins = _include_bin_edge(position[1, :], y_bins, y_edges, side='left')

        return x_bins - 1, y_bins - 1


def compute_bin_events(bins, xbins, ybins=None, occupancy=None):
    """Compute number of events per bin.

    Parameters
    ----------
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    xbins : 1d array
        Bin assignment for the x-dimension for each event.
    ybins : 1d array, optional
        Bin assignment for the y-dimension for each event.
    occupancy : 1d or 2d array, optional
        Occupancy across the spatial bins.
        If provided, used to normalize bin events.

    Returns
    -------
    bin_events : 1d or 2d array
        Amount of events in each bin.
        For 2d, has shape [n_y_bins, n_x_bins] (see notes).

    Notes
    -----
    For the 2D case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.

    Examples
    --------
    Compute the number of events per bin for 1D data, given precomputed xbin for each event:

    >>> bins = 3
    >>> xbins = [0, 2, 1, 0, 1]
    >>> compute_bin_events(bins, xbins)
    array([2, 2, 1])

    Compute the number of events per bin for 2D data, given precomputed x & y bin for each event:

    >>> bins = [2, 2]
    >>> xbins = [0, 0, 0, 1]
    >>> ybins = [0, 0, 1, 1]
    >>> compute_bin_events(bins, xbins, ybins)
    array([[2., 0.],
           [1., 1.]])
    """

    bins = check_position_bins(bins)

    if ybins is None:
        bins = np.arange(0, bins[0] + 1)
        bin_events, _ = np.histogram(xbins, bins=bins)
    else:
        bins = [np.arange(0, bins[0] + 1), np.arange(0, bins[1] + 1)]
        bin_events, _, _ = np.histogram2d(xbins, ybins, bins=bins)
        bin_events = bin_events.T

    if occupancy is not None:
        bin_events = normalize_bin_events(bin_events, occupancy)

    return bin_events


def normalize_bin_events(bin_events, occupancy):
    """Normalize binned events by occupancy.

    Parameters
    ----------
    bin_events : 1d or 2d array
        Spatially binned event counts.
    occupancy : 1d or 2d array
        Spatially binned occupancy.

    Returns
    -------
    normalized_bin_events : 1d or 2d array
        Normalized binned events.

    Notes
    -----
    For any bins in which the occupancy is zero, the output will NaN.

    Examples
    --------
    Normalized a pre-computed 2D binned events array by occupancy:

    >>> bin_events = np.array([[0, 1, 0], [1, 2, 0]])
    >>> occupancy = np.array([[0, 2, 1], [1, 1, 0]])
    >>> normalize_bin_events(bin_events, occupancy)
    array([[nan, 0.5, 0. ],
           [1. , 2. , nan]])
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        normalized_bin_events = bin_events / occupancy

    return normalized_bin_events


def create_position_df(position, timestamps, bins, speed=None, speed_thresh=None, area_range=None):
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
    speed : 1d array
        Current speed for each position.
        Should be the same length as timestamps.
    speed_thresh : float, optional
        Speed threshold to apply.
        If provided, any position values with an associated speed below this value are dropped.
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].

    Returns
    -------
    bindf : pd.DataFrame
        Dataframe representation of position bin information.
    """

    bins = check_position_bins(bins, position)

    data_dict = {'time' : compute_bin_time(timestamps)}
    if speed is not None:
        data_dict['speed'] = speed

    if position.ndim == 1:

        # Spatially bin 1d position data, and collect bin assignment information
        x_edges = compute_bin_edges(position, bins, area_range)
        x_bins = compute_bin_assignment(position, x_edges)
        data_dict['xbin'] = pd.Categorical(\
            x_bins, categories=list(range(0, bins[0])), ordered=True)

    elif position.ndim == 2:

        # Spatially bin 2d position data, and collect bin assignment information
        x_edges, y_edges = compute_bin_edges(position, bins, area_range)
        x_bins, y_bins = compute_bin_assignment(position, x_edges, y_edges)
        data_dict['xbin'] = pd.Categorical(\
            x_bins, categories=list(range(0, bins[0])), ordered=True)
        data_dict['ybin'] = pd.Categorical(\
            y_bins, categories=list(range(0, bins[1])), ordered=True)

    bindf = pd.DataFrame(data_dict)

    if speed_thresh is not None:
        bindf = bindf[bindf.speed > speed_thresh]

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


def compute_occupancy(position, timestamps, bins, speed=None, speed_thresh=None,
                      area_range=None, minimum=None, normalize=False, set_nan=False):
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
    speed : 1d array
        Current speed for each position.
        Should be the same length as timestamps.
    speed_thresh : float, optional
        Speed threshold to apply.
        If provided, any position values with an associated speed below this value are dropped.
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
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

    >>> position = np.array([[1.0, 2.5, 1.5, 3.0, 3.5, 5.0],
    ...                      [5.0, 7.5, 6.5, 5.0, 8.5, 9.0]])
    >>> timestamps = np.linspace(0, 1, position.shape[1])
    >>> compute_occupancy(position, timestamps, bins=[2, 2])
    array([[0.4, 0.2],
           [0.2, 0.2]])
    """

    df = create_position_df(position, timestamps, bins, speed, speed_thresh, area_range)
    occupancy = compute_occupancy_df(df, bins, minimum, normalize, set_nan)

    return occupancy


def _include_bin_edge(position, bin_pos, edges, side='left'):
    """Update bin assignment so last bin includes edge values.

    Parameters
    ----------
    position : 1d array
        Position values.
    bin_pos : 1d array
        The bin assignment for each position.
    edges : 1d array
        The bin edge definitions.
    side : {'left', 'right'}
        Which side was used to compute bin assignment.

    Returns
    -------
    bin_pos : 1d array
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
        bin_pos[mask] = bin_pos[mask] - 1

    elif side == 'right':

        # If side right, left position == edge gets set as 0, so increment by 1
        mask = position == edges[0]
        bin_pos[mask] = bin_pos[mask] + 1

    return bin_pos
