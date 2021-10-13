"""Spatial position and occupancy related functions."""

import numpy as np
import pandas as pd

###################################################################################################
###################################################################################################

def compute_spatial_bin_edges(position, bins, area_range=None):
    """Compute spatial bin edges.

    Parameters
    ----------
    position : 2d array
        Position information across a 2D space.
    bins : list of [int, int]
        The number of bins to divide up the space, defined as [number of x_bins, number of y_bins].
    area_range : list of list
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
        Any values outside of this range will not be used to compute edges.

    Returns
    -------
    x_edges, y_edges : 1d array
        Edge definitions for the spatial binning.
    """

    _, x_edges, y_edges = np.histogram2d(position[0, :], position[1, :],
                                         bins=bins, range=area_range)

    return x_edges, y_edges


def compute_spatial_bin_assignment(position, x_edges, y_edges, include_edge=True):
    """Compute spatial bin assignment.

    Parameters
    ----------
    position : 2d array
        Position information across a 2D space.
    x_edges, y_edges : 1d array
        Edge definitions for the spatial binning.
        Values within the arrays should be monotonically increasing.
    include_edge : bool, optional, default: True
        Whether to include positions on the edge into the bin.

    Returns
    -------
    x_bins, y_bins : 1d array
        Bin assignments for each position.
    """

    x_bins = np.digitize(position[0, :], x_edges, right=True)
    y_bins = np.digitize(position[1, :], y_edges, right=True)

    if include_edge:
        x_bins = _include_bin_edge(x_bins, len(x_edges) - 1)
        y_bins = _include_bin_edge(y_bins, len(y_edges) - 1)

    return x_bins, y_bins


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
        
    >>> timestamp = np.array([0, 10, 30, 60, 80, 90])
    >>> compute_bin_time(timestamp)
    array([10, 20, 30, 20, 10,  0])
    """

    return np.append(np.diff(timestamps), 0)


def compute_occupancy(position, timestamps, bins, speed=None, speed_thresh=5e-6,
                      area_range=None, set_nan=False, normalize=False):
    """Compute occupancy across spatial bin positions.

    Parameters
    ----------
    position : 2d array
        Position information across a 2D space.
    timestamps : 1d array
        Timestamps.
    bins : list of int
        Binning to use for dividing up the space.
    speed : 1d array
        Current speed for each position.
        Should be the same length as timestamps.
    speed_thresh : float, optional
        Speed threshold to apply.
    area_range : list of list
        Edges of the area to bin, defined as [[x_min, x_max], [y_min, y_max]].
    set_nan : bool, optional, default: False
        Whether to set zero occupancy locations as NaN.
    normalize : bool, optional, default: False
        Whether to normalize occupancy to sum to 1.

    Returns
    -------
    2d array
        Occupancy.
    """

    # Compute spatial bins & binning
    x_edges, y_edges = compute_spatial_bin_edges(position, bins, area_range)
    x_bins, y_bins = compute_spatial_bin_assignment(position, x_edges, y_edges)
    bin_time = compute_bin_time(timestamps)

    # Make a temporary pandas dataframe
    df = pd.DataFrame({
        'xbins' : pd.Categorical(x_bins, categories=list(range(bins[0])), ordered=True),
        'ybins' : pd.Categorical(y_bins, categories=list(range(bins[1])), ordered=True),
        'bin_time' : bin_time})

    # Apply the speed threshold (dropping slow / stationary timepoints)
    if np.any(speed):
        df = df[speed > speed_thresh]

    # Group each position into a spatial bin, summing total time spent there
    df = df.groupby(['xbins', 'ybins'])['bin_time'].sum()

    # Extract and re-organize occupancy into 2d array
    occ = np.squeeze(df.values.reshape(*bins, -1)) / 1000

    if normalize:
        occ = occ / np.sum(occ)

    if set_nan:
        occ[occ == 0.] = np.nan

    return occ


def _include_bin_edge(bin_pos, n_bins):
    """Update bin assignment so last bin includes edge values.

    Parameters
    ----------
    bin_pos : 1d array
        The bin assignment for each position.
    n_bins : int
        The number of bins.

    Returns
    -------
    bin_pos : 1d array
        The bin assignment for each position.

    Notes
    -----
    This functions assumes bin assignment done with `right=True`.
    """

    mask = bin_pos == n_bins
    bin_pos[mask] = n_bins - 1

    return bin_pos
