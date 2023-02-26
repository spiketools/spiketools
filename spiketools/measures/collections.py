""""Functions to compute measures across collections of recorded neurons."""

import numpy as np

from spiketools.utils.checks import check_time_bins
from spiketools.measures.spikes import compute_spike_presence

###################################################################################################
###################################################################################################

def detect_empty_time_ranges(all_spikes, bins, time_range=None):
    """Compute the empty time ranges that are common across a collection of recorded neurons.

    Parameters
    ----------
    all_spikes : list of 1d array
        Spike times for a collection of neurons.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the time length of each bin.
        If array, precomputed bin definitions.
    time_range : list of [float, float], optional
        Time range, in seconds, to use to check for empty time ranges.
        Only used if `bins` is a float.

    Returns
    -------
    empty_ranges : list of list of float
        List of ranges indicating time ranges with no spiking across all neurons.

    Examples
    --------
    Detect an empty range across a collection of 2 spiking neurons:

    >>> all_spikes = [np.array([0.25, 0.80, 1.25, 1.75, 4.15, 4.95]),
    ...               np.array([0.15, 0.55, 1.55, 1.95, 4.35, 4.85])]
    >>> detect_empty_time_ranges(all_spikes, bins=1, time_range=[0, 5])
    [[2, 4]]
    """

    bin_emptiness = find_empty_bins(all_spikes, bins, time_range)
    empty_ranges = find_empty_ranges(bin_emptiness, bins, time_range)

    return empty_ranges


def find_empty_bins(all_spikes, bins, time_range=None):
    """Find empty time bins across a collection of recorded neurons.

    Parameters
    ----------
    all_spikes : list of 1d array
        Spike times for a collection of neurons.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the time length of each bin.
        If array, precomputed bin definitions.
    time_range : list of [float, float], optional
        Time range, in seconds, to use to check for empty time bins.
        Only used if `bins` is a float.

    Returns
    -------
    bin_emptiness : 1d array
        Boolean array indicating which bins are empty across all neurons.
    """

    bins = check_time_bins(bins, time_range)

    all_presences = np.full([len(all_spikes), len(bins)-1], False)
    for ind, cur_spikes in enumerate(all_spikes):
        all_presences[ind, :] = compute_spike_presence(cur_spikes, bins)

    bin_emptiness = np.all(~all_presences, 0)

    return bin_emptiness


def find_empty_ranges(bin_emptiness, bins, time_range=None):
    """Find empty time ranges based on time bin emptiness for a collection of recorded neurons.

    Parameters
    ----------
    bin_emptiness : 1d array
        Boolean array indicating which bins are empty across all neurons.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the time length of each bin.
        If array, precomputed bin definitions.
    time_range : list of [float, float], optional
        Time range, in seconds, to use to check for empty time ranges.
        Only used if `bins` is a float.

    Returns
    -------
    empty_ranges : list of list of float
        List of ranges indicating time ranges with no spiking across all neurons.
    """

    bins = check_time_bins(bins, time_range)

    diff_inds = np.where(np.diff(bin_emptiness))[0] + 1

    empty_ranges = []

    # Special case: first bin is empty
    if bin_emptiness[0] == True:
        empty_ranges.append([bins[0], bins[diff_inds[0]]])
        diff_inds = np.delete(diff_inds, 0)

    # Special case: last bin is empty
    if bin_emptiness[-1] == True:
        empty_ranges.append([bins[diff_inds[-1]], bins[-1]])
        diff_inds = np.delete(diff_inds, -1)

    # Over empty regions can be found be stepping across pairs of diffs (changes)
    diff_inds_2d = np.reshape(diff_inds, [int(len(diff_inds) / 2), 2])
    empty_ranges.extend([[bins[inds[0]], bins[inds[1]]] for inds in diff_inds_2d])

    empty_ranges.sort(key=lambda x: x[0])

    return empty_ranges
