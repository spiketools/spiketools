"""Spatial target related functions."""

import numpy as np

from spiketools.spatial.occupancy import normalize_bin_counts
from spiketools.spatial.checks import check_position_bins

###################################################################################################
###################################################################################################

def compute_target_bins(target_frs, bins, xbins, ybins=None, target_occupancy=None):
    """Compute binned firing based on spatial target.

    Parameters
    ----------
    target_frs : 2d array
        The firing rate per target segment, organized as [n_trials, n_targets_per_trial].
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    xbins, ybins : 1d array
        The bin assignments per target. `ybins` is optional if using 1d binning.
        The length should equal the length of flattened `target_frs`.
    target_occupancy : 1d or 2d array, optional
        The number of targets per spatial target bin.
        If provided, used to normalize the spiking activity.

    Returns
    -------
    target_bins : 1d or 2d array
        The target related spiking activity.

    Notes
    -----
    For the 2D case, note that while the inputs to this function list the x-axis first,
    the output of this function, being a 2d array, follows the numpy convention in which
    columns (y-axis) are on the 0th dimension, and rows (x-axis) are on the 1th dimension.
    """

    bins = check_position_bins(bins)

    target_bins = np.zeros(np.flip(bins))

    if target_bins.ndim == 1:
        for xb, fr in zip(xbins, target_frs.flatten()):
            target_bins[xb] += fr

    elif target_bins.ndim == 2:
        for xb, yb, fr in zip(xbins, ybins, target_frs.flatten()):
            target_bins[yb, xb] += fr

    if target_occupancy is not None:
        target_bins = normalize_bin_counts(target_bins, target_occupancy)

    return target_bins
