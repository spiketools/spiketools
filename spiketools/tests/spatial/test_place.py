"""Tests for spiketools.spatial.place"""

import numpy as np

from spiketools.spatial.occupancy import compute_trial_occupancy

from spiketools.spatial.place import *

###################################################################################################
###################################################################################################

def test_compute_place_bins():

    spikes = np.array([0.1, 0.65, 1.1, 1.6, 1.9, 2.4, 2.9, 3.3, 3.77, 4.4])
    position = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                         [0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5]])
    timestamps = np.arange(0, 5.5, 0.5)
    bins = [2, 3]

    place_bins = compute_place_bins(spikes, position, timestamps, bins)
    assert isinstance(place_bins, np.ndarray)
    assert np.array_equal(place_bins.shape, np.flip(bins))
    expected = np.array([[5, 0], [0, 2], [0, 3]])
    assert np.array_equal(place_bins, expected)

    # Check with speed dropping
    speed = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    speed_threshold = 0.5
    place_bins = compute_place_bins(spikes, position, timestamps, bins,
                                    speed=speed, speed_threshold=speed_threshold)
    expected = np.array([[3, 0], [0, 2], [0, 2]])
    assert np.array_equal(place_bins, expected)

def test_compute_trial_place_bins():

    spikes = np.array([0.1, 0.65, 1.1, 1.6, 1.9, 2.4, 2.9, 3.3, 3.77, 4.4])
    position = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                         [0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5]])
    timestamps = np.arange(0, 5.5, 0.5)
    start_times = np.array([0, 2.01])
    stop_times = np.array([2.01, 4.51])
    bins = [2, 3]

    place_bins_trial = compute_trial_place_bins(spikes, position, timestamps, bins,
                                                start_times, stop_times)
    assert isinstance(place_bins_trial, np.ndarray)
    assert np.array_equal(place_bins_trial.shape, np.array([len(start_times), bins[1], bins[0]]))
    expected = np.array([[[0, 0], [0, 5], [0, 0]],
                         [[0, 2], [0, 0], [0, 3]]])
    assert np.array_equal(place_bins_trial, expected)

    # Check with speed dropping
    speed = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    speed_threshold = 0.5
    place_bins_trial = compute_trial_place_bins(spikes, position, timestamps, bins,
                                                start_times, stop_times,
                                                speed=speed, speed_threshold=speed_threshold)
    expected = np.array([[[0, 0], [0, 3], [0, 0]],
                         [[0, 2], [0, 0], [0, 2]]])
    assert np.array_equal(place_bins_trial, expected)

    # Check with occupancy normalization
    trial_occupancy = compute_trial_occupancy(position, timestamps, bins, start_times, stop_times)
    place_bins_trial = compute_trial_place_bins(spikes, position, timestamps, bins,
                                                start_times, stop_times,
                                                trial_occupancy=trial_occupancy)
    expected = np.array([[[np.nan, np.nan], [np.nan, 2.5], [np.nan, np.nan]],
                         [[np.nan, 2], [np.nan, np.nan], [np.nan, 3]]])
    assert np.array_equal(place_bins_trial, expected, equal_nan=True)
