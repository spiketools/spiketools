"""Place cell related functions."""

import numpy as np

from spiketools.spatial.utils import compute_nbins
from spiketools.spatial.occupancy import compute_occupancy, compute_bin_counts_pos
from spiketools.utils.extract import (get_range, get_values_by_time_range, get_values_by_times,
                                      threshold_spikes_by_values)

###################################################################################################
###################################################################################################

def compute_place_bins(spikes, position, timestamps, bins, area_range=None,
                       speed=None, speed_threshold=None, time_threshold=None,
                       occupancy=None):
    """Compute the spatially binned spiking activity.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
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
    occupancy : 1d or 2d array, optional
        Computed occupancy across the space.
        If provided, used to normalize bin counts.

    Returns
    -------
    place_bins : 2d array
        The spike activity per spatial bin.
    """

    if speed is not None:
        spikes = threshold_spikes_by_values(\
            spikes, timestamps, speed, speed_threshold, time_threshold)

    spike_positions = get_values_by_times(timestamps, position, spikes, time_threshold)
    place_bins = compute_bin_counts_pos(spike_positions, bins, area_range, occupancy)

    return place_bins


def compute_trial_place_bins(spikes, position, timestamps, bins, trial_starts, trial_stops,
                             area_range=None, speed=None, speed_threshold=None,
                             time_threshold=None, normalize=True, flatten=False,
                             **occupancy_kwargs):
    """Compute the spatially binned spiking activity, across trials.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    position : 1d or 2d array
        Position values.
    timestamps : 1d array
        Timestamps.
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    trial_starts : 1d array
        The start times of each trial.
    trial_stops : 1d array
        The stop times of each trial.
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
    normalize : bool, optional, default: True
        Whether to compute trial-level occupancy and use to normalize spatially binned firing.
    flatten : bool, optional, default: False
        Whether the flatten the spatial bins per trial. Only used if position data are 2d.
    occupancy_kwargs
        Additional arguments to pass into the the `compute_occupancy` function.

    Returns
    -------
    place_bins_trial : 1d or 2d or 3d array
        The spike activity per spatial bin, per trial.
        If `flatten` is True, for a 2d position input, the output is 2d, as [n_trial, n_bins].
        Otherwise, for a 2d position input, the output is 3d, as [n_trials, n_ybins, n_xbins].
    """

    t_occ = None
    t_speed = None
    place_bins_trial = np.zeros([len(trial_starts), *np.flip(bins)])
    for ind, (start, stop) in enumerate(zip(trial_starts, trial_stops)):

        t_spikes = get_range(spikes, start, stop)
        t_times, t_pos = get_values_by_time_range(timestamps, position, start, stop)
        if speed is not None:
            _, t_speed = get_values_by_time_range(timestamps, speed, start, stop)

        if normalize:
            t_occ = compute_occupancy(t_pos, t_times, bins, area_range,
                                      t_speed, speed_threshold, time_threshold,
                                      **occupancy_kwargs)

        place_bins_trial[ind, :, :] = compute_place_bins(t_spikes, t_pos, t_times, bins, area_range,
                                                         t_speed, speed_threshold, time_threshold,
                                                         t_occ)

    if flatten:
        place_bins_trial = np.reshape(place_bins_trial, [len(trial_starts), compute_nbins(bins)])

    return place_bins_trial
