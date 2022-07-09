"""Tests for spiketools.utils.trials"""

import numpy as np

from spiketools.utils.trials import *

###################################################################################################
###################################################################################################

def test_epoch_spikes_by_event():

    spikes = np.array([2.5, 3.5, 4.25, 5.5, 6.1, 8., 9.25, 9.75, 10.5, 12., 14.1, 15.2, 15.9])
    events = np.array([5, 10, 15])
    window = [-1, 1]

    trials = epoch_spikes_by_event(spikes, events, window)
    assert isinstance(trials, list)
    assert isinstance(trials[0], np.ndarray)
    assert len(trials) == len(events)
    assert np.array_equal(trials[0], np.array([4.25, 5.5]) - events[0])
    assert np.array_equal(trials[1], np.array([9.25, 9.75, 10.5]) - events[1])
    assert np.array_equal(trials[2], np.array([14.1, 15.2, 15.9]) - events[2])

def test_epoch_spikes_by_range():

    spikes = np.array([2.5, 3.5, 4.25, 5.5, 6.1, 8., 9.25, 9.75, 10.5, 12., 14.1, 15.2, 15.9])
    starts = np.array([5, 12])
    stops = np.array([10, 15])

    trials = epoch_spikes_by_range(spikes, starts, stops)
    assert isinstance(trials, list)
    assert isinstance(trials[0], np.ndarray)
    assert len(trials) == len(starts)
    assert np.array_equal(trials[0], np.array([5.5, 6.1, 8., 9.25, 9.75]))
    assert np.array_equal(trials[1], np.array([12., 14.1]))

    # Check with time reseting
    trials = epoch_spikes_by_range(spikes, starts, stops, reset=True)
    assert np.array_equal(trials[0], np.array([5.5, 6.1, 8., 9.25, 9.75]) - starts[0])
    assert np.array_equal(trials[1], np.array([12., 14.1]) - starts[1])

def test_epoch_data_by_time():

    times = np.array([2.5, 3.5, 4.25, 5.5, 6.1, 8., 9.25, 9.75, 10.5, 12., 14.1, 15.2, 15.9])
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    timepoints = np.array([5, 10, 15])

    tvalues = epoch_data_by_time(times, values, timepoints)
    assert isinstance(tvalues, list)
    assert len(tvalues) == len(timepoints)
    assert np.array_equal(tvalues, np.array([3, 7, 11]))

def test_epoch_data_by_event():

    times = np.array([2.5, 3.5, 4.25, 5.5, 6.1, 8., 9.25, 9.75, 10.5, 12., 14.1, 15.2, 15.9])
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    events = np.array([5, 10, 15])
    window = [-1, 1]

    ttimes, tvalues = epoch_data_by_event(times, values, events, window)
    assert isinstance(ttimes, list)
    assert isinstance(tvalues, list)
    assert isinstance(ttimes[0], np.ndarray)
    assert isinstance(tvalues[0], np.ndarray)
    assert len(ttimes) == len(tvalues) == len(events)

    assert np.array_equal(ttimes[0], np.array([4.25, 5.5]) - events[0])
    assert np.array_equal(tvalues[0], np.array([2, 3]))
    assert np.array_equal(ttimes[1], np.array([9.25, 9.75, 10.5]) - events[1])
    assert np.array_equal(tvalues[1], np.array([6, 7, 8]))
    assert np.array_equal(ttimes[2], np.array([14.1, 15.2, 15.9]) - events[2])
    assert np.array_equal(tvalues[2], np.array([10, 11, 12]))

def test_epoch_data_by_range():

    times = np.array([2.5, 3.5, 4.25, 5.5, 6.1, 8., 9.25, 9.75, 10.5, 12., 14.1, 15.2, 15.9])
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    starts = np.array([5, 12])
    stops = np.array([10, 15])

    ttimes, tvalues = epoch_data_by_range(times, values, starts, stops)
    assert isinstance(ttimes, list)
    assert isinstance(tvalues, list)
    assert isinstance(ttimes[0], np.ndarray)
    assert isinstance(tvalues[0], np.ndarray)
    assert len(ttimes) == len(tvalues) == len(starts)

    assert np.array_equal(ttimes[0], np.array([5.5, 6.1, 8., 9.25, 9.75]))
    assert np.array_equal(tvalues[0], np.array([3, 4, 5, 6, 7]))
    assert np.array_equal(ttimes[1], np.array([12.0, 14.1]))
    assert np.array_equal(tvalues[1], np.array([9, 10]))

    # Check with time reseting
    ttimes, tvalues = epoch_data_by_range(times, values, starts, stops, reset=True)
    assert np.array_equal(ttimes[0], np.array([5.5, 6.1, 8., 9.25, 9.75]) - starts[0])
    assert np.array_equal(tvalues[0], np.array([3, 4, 5, 6, 7]))
    assert np.array_equal(ttimes[1], np.array([12.0, 14.1]) - starts[1])
    assert np.array_equal(tvalues[1], np.array([9, 10]))
