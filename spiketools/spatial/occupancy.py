"""Spatial position and occupancy related functions."""

import warnings

import numpy as np
import pandas as pd

###################################################################################################
###################################################################################################

def compute_nbins(bins):
    """Compute the number of bins for a given bin definition.

    Parameters
    ----------
    bins : list of [int, int]
        Bin definition.

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

    return bins[0] * bins[1]


def compute_bin_edges(position, bins, area_range=None):
    """Compute spatial bin edges.

    Parameters
    ----------
    position : 1d or 2d array
        Position values across a 1D or 2D space.
    bins : list of [int] or [int, int]
        The number of bins to divide up the space, defined as [number of x_bins, number of y_bins].
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

    if position.ndim == 1:
        _, x_edges = np.histogram(position, bins=bins[0], range=area_range)

        return x_edges

    elif position.ndim == 2:
        _, x_edges, y_edges = np.histogram2d(position[0, :], position[1, :],
                                             bins=bins, range=area_range)
        return x_edges, y_edges

    else:
        raise ValueError('Position input should be 1d or 2d.')


def compute_bin_assignment(position, x_edges, y_edges=None, include_edge=True):
    """Compute spatial bin assignment.

    Parameters
    ----------
    position : 1d or 2d array
        Position information across a 1D or 2D space.
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

    warning = "There are position values outside of the given bin ranges."

    if position.ndim == 1:
        x_bins = np.digitize(position, x_edges, right=False)

        if include_edge:
            x_bins = _include_bin_edge(position, x_bins, x_edges, side='left')

        if np.any(x_bins == 0) or np.any(x_bins == len(x_edges)):
            warnings.warn(warning)

        return x_bins - 1

    elif position.ndim == 2:
        x_bins = np.digitize(position[0, :], x_edges, right=False)
        y_bins = np.digitize(position[1, :], y_edges, right=False)

        if include_edge:
            x_bins = _include_bin_edge(position[0, :], x_bins, x_edges, side='left')
            y_bins = _include_bin_edge(position[1, :], y_bins, y_edges, side='left')

        x_check = np.any(x_bins == 0) or np.any(x_bins == len(x_edges))
        y_check = np.any(y_bins == 0) or np.any(y_bins == len(y_edges))
        if x_check or y_check:
            warnings.warn(warning)

        return x_bins - 1, y_bins - 1

    else:
        raise ValueError('Position input should be 1d or 2d.')


def compute_bin_firing(bins, xbins, ybins=None, occupancy=None, transpose=True):
    """Compute firing per bin, given the bin assignment of each spike.

    Parameters
    ----------
    bins : list of [int] or [int, int]
        Bin definition.
    xbins : 1d array
        Bin assignment for the x-dimension for each spike.
    ybins : 1d array, optional
        Bin assignment for the y-dimension for each spike.
    occupancy : 1d or 2d array, optional
        Occupancy across the spatial bins.
        If provided, used to normalize bin firing.
    transpose : bool, optional, default: True
        Whether to transpose the output, so that x-bins lie on the x-axis of the array.

    Returns
    -------
    bin_firing : 2d array
        Amount of firing in each bin.

    Examples
    --------
    Compute the amount of firing per bin, given precomputed x & y bin for each spike:

    >>> bins = [2, 2]
    >>> xbins = [0, 0, 0, 1]
    >>> ybins = [0, 0, 1, 1]
    >>> compute_bin_firing(bins, xbins, ybins)
    array([[2., 0.],
           [1., 1.]])
    """

    if ybins is None:
        bin_firing = np.histogram(xbins, bins=np.arange(0, bins[0] + 1))[0]
    else:
        bin_firing = np.histogram2d(xbins, ybins, bins=[np.arange(0, bl + 1) for bl in bins])[0]

    if transpose:
        bin_firing = bin_firing.T

    if occupancy is not None:
        bin_firing = normalize_bin_firing(bin_firing, occupancy)

    return bin_firing


def normalize_bin_firing(bin_firing, occupancy):
    """Normalize binned firing by occupancy.

    Parameters
    ----------
    bin_firing : 1d or 2d array
        Spatially binned firing.
    occupancy : 1d or 2d array
        Spatially binned occupancy.

    Returns
    -------
    normalized_bin_firing : 1d or 2d array
        Normalized binned firing.

    Notes
    -----
    For any bins in which the occupancy is zero, the output will NaN.

    Examples
    --------
    Normalized a pre-computed 2D binned firing array by occupancy:

    >>> bin_firing = np.array([[0, 1, 0], [1, 2, 0]])
    >>> occupancy = np.array([[0, 2, 1], [1, 1, 0]])
    >>> normalize_bin_firing(bin_firing, occupancy)
    array([[nan, 0.5, 0. ],
           [1. , 2. , nan]])
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        normalized_bin_firing = bin_firing / occupancy

    return normalized_bin_firing


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


def compute_occupancy(position, timestamps, bins, speed=None, speed_thresh=None, minimum=None,
                      normalize=False, set_nan=False, area_range=None, transpose=True):
    """Compute occupancy across spatial bin positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position information across a 1D or 2D space.
    timestamps : 1d array
        Timestamps.
    bins : list of [int] or [int, int]
        Binning to use for dividing up the space.
    speed : 1d array
        Current speed for each position.
        Should be the same length as timestamps.
    speed_thresh : float, optional
        Speed threshold to apply.
    minimum : float, optional
        The minimum required occupancy.
        If defined, any values below this are set to zero.
    normalize : bool, optional, default: False
        Whether to normalize occupancy to sum to 1.
    set_nan : bool, optional, default: False
        Whether to set zero occupancy locations as NaN.
    area_range : list of list, optional
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
    transpose : bool, optional, default: True
        Whether to transpose the output, so that x-bins lie on the x-axis of the array.

    Returns
    -------
    occupancy : 1d or 2d array
        Computed occupancy across the space.

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

    # Initialize dictionary to collect data, adding bin time information
    data_dict = {'bin_time' : compute_bin_time(timestamps)}

    if position.ndim == 1:

        # Spatially bin 1d position data, and collect into a dictionary
        x_edges = compute_bin_edges(position, bins, area_range)
        x_bins = compute_bin_assignment(position, x_edges)

        # Add binned space information to data dictionary, and define how to group data
        data_dict['xbins'] = pd.Categorical(\
            x_bins, categories=list(range(0, bins[0])), ordered=True)
        groupby = ['xbins']

    elif position.ndim == 2:

        # Spatially bin 1d position data, and collect into a dictionary
        x_edges, y_edges = compute_bin_edges(position, bins, area_range)
        x_bins, y_bins = compute_bin_assignment(position, x_edges, y_edges)

        # Add binned space information to data dictionary, and define how to group data
        data_dict['xbins'] = pd.Categorical(\
            x_bins, categories=list(range(0, bins[0])), ordered=True)
        data_dict['ybins'] = pd.Categorical(\
            y_bins, categories=list(range(0, bins[1])), ordered=True)
        groupby = ['xbins', 'ybins']

    else:
        raise ValueError('Position input should be 1d or 2d.')

    # Collect information together into a temporary dataframe
    df = pd.DataFrame(data_dict)

    # Apply the speed threshold (dropping slow / stationary timepoints)
    if np.any(speed):
        df = df[speed > speed_thresh]

    # Group each position into a spatial bin, summing total time spent there
    df = df.groupby(groupby)['bin_time'].sum()

    # Extract and re-organize occupancy into array
    occupancy = np.squeeze(df.values.reshape(*bins, -1))

    if minimum:
        occupancy[occupancy < minimum] = 0.

    if normalize:
        occupancy = occupancy / np.sum(occupancy)

    if set_nan:
        occupancy[occupancy == 0.] = np.nan

    if transpose:
        occupancy = occupancy.T

    return occupancy


def _include_bin_edge(position, bin_pos, edges, side='left'):
    """Update bin assignment so last bin includes edge values.

    Parameters
    ----------
    position : 1d array
        The position values.
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

    if side == 'left':

        # If side left, right position == edge gets set as len(bins), so decrement by 1
        mask = position == edges[-1]
        bin_pos[mask] = bin_pos[mask] - 1

    elif side == 'right':

        # If side right, left position == edge gets set as 0, so increment by 1
        mask = position == edges[0]
        bin_pos[mask] = bin_pos[mask] + 1

    else:
        raise ValueError("Input for 'side' not understood.")

    return bin_pos
