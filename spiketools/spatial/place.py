"""Place cell related functions."""

import numpy as np

from spiketools.spatial.utils import compute_nbins
from spiketools.spatial.occupancy import compute_bin_counts_pos
from spiketools.spatial.checks import check_bin_definition
from spiketools.utils.checks import check_array_orientation
from spiketools.utils.extract import (get_range, get_values_by_time_range, get_values_by_times,
                                      threshold_spikes_by_values)

###################################################################################################
###################################################################################################

def compute_place_bins(spikes, position, timestamps, bins, area_range=None,
                       speed=None, speed_threshold=None, time_threshold=None,
                       occupancy=None, orientation=None):
    """Compute the spatially binned spiking activity.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
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
    speed_threshold : float, optional
        Speed threshold to apply.
        If provided, any position values with an associated speed below this value are dropped.
    time_threshold : float, optional
        A maximum time threshold, per bin observation, to apply.
        If provided, any bin values with an associated time length above this value are dropped.
    occupancy : 1d or 2d array, optional
        Computed occupancy across the space.
        If provided, used to normalize bin counts.
    orientation : {'row', 'column'}, optional
        The orientation of the position data.
        If not provided, is inferred from the position data.

    Returns
    -------
    place_bins : 2d array
        The spike activity per spatial bin.

    Examples
    --------
    Compute spike activity across 2d spatial bins:

    >>> spikes = np.array([0.2, 0.25, 0.3, 0.38, 0.41, 0.5, 0.59, 0.77, 0.95, 0.96])
    >>> position = np.array([[0.1, 0.3, 0.35, 0.36, 0.37, 0.4, 0.45, 0.46, 0.55, 0.7],
    ...                      [1.0, 1.5, 1.55, 1.65, 1.66, 2.0, 3.0, 4.0, 5.5, 7.0]])
    >>> timestamps = np.array([0.01, 0.03, 0.2, 0.25, 0.45, 0.46, 0.47, 0.49, 0.5, 0.65])
    >>> bins = [3, 2]
    >>> compute_place_bins(spikes, position, timestamps, bins)
    array([[5, 0, 0],
           [0, 1, 4]])
    """

    if speed is not None:
        spikes = threshold_spikes_by_values(\
            spikes, timestamps, speed, speed_threshold, time_threshold)

    spike_positions = get_values_by_times(timestamps, position, spikes, time_threshold)
    place_bins = compute_bin_counts_pos(spike_positions, bins, area_range, occupancy, orientation)

    return place_bins


def compute_trial_place_bins(spikes, position, timestamps, bins, start_times, stop_times,
                             area_range=None, speed=None, speed_threshold=None, time_threshold=None,
                             trial_occupancy=None, flatten=False, orientation=None):
    """Compute the spatially binned spiking activity, across trials.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
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
    speed_threshold : float, optional
        Speed threshold to apply.
        If provided, any position values with an associated speed below this value are dropped.
    time_threshold : float, optional
        A maximum time threshold, per bin observation, to apply.
        If provided, any bin values with an associated time length above this value are dropped.
    trial_occupancy : 2d or 3d array, optional
        Computed occupancy across the space, across trials.
        If provided, used to normalize bin counts per trial.
    flatten : bool, optional, default: False
        Whether the flatten the spatial bins per trial. Only used if position data are 2d.
    orientation : {'row', 'column'}, optional
        The orientation of the position data.
        If not provided, is inferred from the position data.

    Returns
    -------
    place_bins_trial : 1d or 2d or 3d array
        The spike activity per spatial bin, per trial.
        If `flatten` is True, for a 2d position input, the output is 2d, as [n_trial, n_bins].
        Otherwise, for a 2d position input, the output is 3d, as [n_trials, n_ybins, n_xbins].

    Examples
    --------
    Compute spike activity, in 1d spatial bins across 2 trials:

    >>> spikes = np.array([0.15, 0.22, 0.28, 0.41, 0.50, 0.65, 0.77, 0.81, 0.95])
    >>> position = np.array([1.0, 3.5, 2.0, 1.5, 3.0, 3.5, 4.0, 5.0, 3.5, 2.5])
    >>> timestamps = np.array([0.10, 0.20, 0.25, 0.35, 0.45, 0.55, 0.6, 0.7, 0.80, 0.95])
    >>> bins = 2
    >>> start_times, stop_times = np.array([0, 0.6]), np.array([0.4, 1.0])
    >>> compute_trial_place_bins(spikes, position, timestamps, bins, start_times, stop_times)
    array([[2., 1.],
           [3., 1.]])

    Compute spike activity across trials, normalizing by trial-level occupancy:

    >>> from spiketools.spatial.occupancy import compute_trial_occupancy
    >>> trial_occ = compute_trial_occupancy(position, timestamps, bins, start_times, stop_times)
    >>> compute_trial_place_bins(spikes, position, timestamps, bins,
    ...                          start_times, stop_times, trial_occupancy=trial_occ)
    array([[10., 20.],
           [20.,  5.]])
    """

    t_occ = None
    t_speed = None

    bins = check_bin_definition(bins, position)
    orientation = check_array_orientation(position, len(bins)) if not orientation else orientation

    place_bins_trial = np.zeros([len(start_times), *np.flip(bins)])
    for ind, (start, stop) in enumerate(zip(start_times, stop_times)):

        t_spikes = get_range(spikes, start, stop)
        t_times, t_pos = get_values_by_time_range(timestamps, position, start, stop)
        if speed is not None:
            _, t_speed = get_values_by_time_range(timestamps, speed, start, stop)

        if trial_occupancy is not None:
            t_occ = trial_occupancy[ind, :]

        place_bins_trial[ind, :] = compute_place_bins(t_spikes, t_pos, t_times, bins, area_range,
                                                      t_speed, speed_threshold, time_threshold,
                                                      t_occ, orientation)

    if flatten:
        place_bins_trial = np.reshape(place_bins_trial, [len(start_times), compute_nbins(bins)])

    return place_bins_trial
