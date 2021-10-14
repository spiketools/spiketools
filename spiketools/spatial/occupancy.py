"""Spatial position and occupancy related functions."""

import numpy as np
import pandas as pd

###################################################################################################
###################################################################################################

def compute_spatial_bin_edges(position, bins):
    """Compute spatial bin edges.

    Parameters
    ----------
    position : 2d array
        Position information across a 2D space.
    bins : list of [int, int]
        The number of bins to divide up the space, defined as [number of x_bins, number of y_bins]. 

    Returns
    -------
    x_edges, y_edges : 1d array
        Edge definitions for the spatial binning.
	
    Examples
    --------
    Compute bin edges for an example rectangular field, with x-range values of 1 - 5 & y-range values of 6 - 10:
    So, position points are: (1, 6), (2, 7), (3, 8), (4, 9), (5, 10).

    >>> position = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> bins = [5, 4]
    >>> compute_spatial_bin_edges(position, bins)
    (array([1. , 1.8, 2.6, 3.4, 4.2, 5. ]), array([ 6.,  7.,  8.,  9., 10.]))
    """

    _, x_edges, y_edges = np.histogram2d(position[0, :], position[1, :], bins=bins)

    return x_edges, y_edges


def compute_spatial_bin_assignment(position, x_edges, y_edges):
    """Compute spatial bin assignment.

    Parameters
    ----------
    position : 2d array
        Position information across a 2D space.
    x_edges, y_edges : 1d array
        Edge definitions for the spatial binning.

    Returns
    -------
    x_bins, y_bins : 1d array
        Bin assignments for each position.
    """
    # replace pd.cut with np.digitize 
    x_bins = np.digitize(position[0, :], x_edges, right=True)
    y_bins = np.digitize(position[1, :], y_edges, right=True)
    
    return x_bins, y_bins


def compute_bin_width(timestamps):
    """Compute bin width.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.

    Returns
    -------
    1d array
        Width, in time, of each bin.
    """

    return np.append(np.diff(timestamps), 0)


def compute_occupancy(position, timestamps, bins, speed=None, speed_thresh=5e-6, set_nan=False):
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
    set_nan : bool, optional, default: False
        Whether to set zero occupancy locations as NaN.

    Returns
    -------
    2d array
        Occupancy.
    """

    x_edges, y_edges = compute_spatial_bin_edges(position, bins)
    x_bins, y_bins = compute_spatial_bin_assignment(position, x_edges, y_edges)
    bin_width = compute_bin_width(timestamps)

    # TODO: refactor & drop DF here
    # Make a temporary pandas dataframe
    df = pd.DataFrame()
    df['xbins'] = x_bins
    df['ybins'] = y_bins
    df['bin_width'] = bin_width

    if np.any(speed):
        df = df[speed > speed_thresh]

    df = df.groupby(['xbins', 'ybins'])['bin_width'].sum()

    # occupancy time
    occ = np.squeeze(df.values.reshape(*bins, -1)) / 1000

    if set_nan:
        occ[occ == 0.] = np.nan

    return occ
