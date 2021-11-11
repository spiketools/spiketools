"""Tests for spiketools.spatial.occupancy"""

import numpy as np

from spiketools.spatial.occupancy import *

###################################################################################################
###################################################################################################

def test_compute_spatial_bin_edges():

    # checks for two inputs of the same size, different number of bins
    position = np.array([[1., 2., 3., 4.], [0., 1., 2., 3.]])
    bins = [4, 3]
    x_edges, y_edges = compute_spatial_bin_edges(position, bins)

    # dimension checks
    assert len(x_edges) == bins[0]+1
    assert len(y_edges) == bins[1]+1

    # first and last element checks
    assert x_edges[0] == min(position[0, :])
    assert x_edges[-1] == max(position[0, :])
    assert y_edges[0] == min(position[1, :])
    assert y_edges[-1] == max(position[1, :])

    # sorting check
    assert bool((np.sort(x_edges) == x_edges).sum())
    assert bool((np.sort(y_edges) == y_edges).sum())

    # checks for two inputs such that one is the other one shuffled
    position = np.array([[1., 2., 3., 4.], [4., 1., 3., 2.]])
    bins = [4, 4]
    x_edges, y_edges = compute_spatial_bin_edges(position, bins)

    # check that bins are the same
    assert np.sum(x_edges == y_edges) == bins[0] + 1

    # test for regular input (x) and all zeros input (y)
    position = np.array([[1., 2., 3., 4.], [0., 0., 0., 0.]])
    x_edges, y_edges = compute_spatial_bin_edges(position, bins)

    # all zeros case check
    assert np.sum(y_edges == np.linspace(-0.5, 0.5, bins[1] + 1)) == bins[1] + 1

def test_compute_spatial_bin_assignment():

    # test 1: test with simple data
    position = np.array([[1, 3, 5, 7], [1, 3, 5, 7]])
    x_edges = np.array([0, 2, 4, 6, 8])
    y_edges = np.array([0, 2, 4, 6, 8])

    # test for simple input (position, x_edges, y_edges)
    x_bins, y_bins = compute_spatial_bin_assignment(position, x_edges, y_edges)

    # output type check
    assert isinstance(x_bins, np.ndarray)
    assert isinstance(y_bins, np.ndarray)

    # output dimension check
    assert x_bins.all() == y_bins.all()
    assert position[0].shape == x_bins.shape
    assert position[1].shape == y_bins.shape

    # test 2: test with more random data
    position = np.random.uniform(0, 2, (2, 10))
    x_edges = np.arange(0, 2.2, 0.2)
    y_edges = np.arange(0, 2.2, 0.2)

    # test for input position (randomly generated) and x_edges, y_edges
    x_bins, y_bins = compute_spatial_bin_assignment(position, x_edges, y_edges)

    # output type check
    assert isinstance(x_bins, np.ndarray)
    assert isinstance(y_bins, np.ndarray)

    # output dimension check
    assert x_bins.all() == y_bins.all()
    assert position[0].shape == x_bins.shape
    assert position[1].shape == y_bins.shape

def test_compute_bin_time():

    # define a timestamp, with irregular times
    timestamp = np.array([0, 10, 20, 30, 45, 50, 60, 70, 80, 90, 120])
    bin_time = compute_bin_time(timestamp)

    # check dimensions & sum
    assert bin_time.shape[0] == timestamp.shape[0]
    assert np.sum(np.diff(timestamp)) == np.sum(bin_time)

def test_compute_occupancy():

    # define a position, timestamp, and bins
    position = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    timestamp = np.linspace(0, 100000, position.shape[1])
    bins = [4, 3]
    occ = compute_occupancy(position, timestamp, bins)

    # check dimensions & sum (should be the same if binning is swapped)
    assert occ.shape[0] == bins[0]
    assert occ.shape[1] == bins[1]
    assert np.nansum(occ) == np.nansum(compute_occupancy(position, timestamp, [bins[1], bins[0]]))
