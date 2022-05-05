"""Tests for spiketools.spatial.occupancy"""

import numpy as np

from pytest import warns

from spiketools.tests.tsettings import BINS

from spiketools.spatial.occupancy import *

###################################################################################################
###################################################################################################

def test_compute_nbins():

    out = compute_nbins([5, 5])
    assert out == 25

def test_compute_spatial_bin_edges():

    # checks for two inputs of the same size, different number of bins
    position = np.array([[1., 2., 3., 4.], [0., 1., 2., 3.]])
    x_edges, y_edges = compute_spatial_bin_edges(position, BINS)

    # dimension checks
    assert len(x_edges) == BINS[0]+1
    assert len(y_edges) == BINS[1]+1

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
    x_edges, y_edges = compute_spatial_bin_edges(position, BINS)

    # check that bin ranges are the same
    assert x_edges[0] == y_edges[0]
    assert x_edges[-1] == y_edges[-1]

    # test for regular input (x) and all zeros input (y)
    position = np.array([[1., 2., 3., 4.], [0., 0., 0., 0.]])
    x_edges, y_edges = compute_spatial_bin_edges(position, BINS)

    # all zeros case check
    assert np.sum(y_edges == np.linspace(-0.5, 0.5, BINS[1] + 1)) == BINS[1] + 1

def test_compute_spatial_bin_assignment():

    # test with simple data, checking accuracy
    position = np.array([[1, 3, 5, 7], [1, 3, 5, 7]])
    x_edges = np.array([0, 2, 4, 6, 8])
    y_edges = np.array([0, 2, 4, 6, 8])
    x_bins, y_bins = compute_spatial_bin_assignment(position, x_edges, y_edges)
    expected = np.array([0, 1, 2, 3])
    assert isinstance(x_bins, np.ndarray)
    assert isinstance(y_bins, np.ndarray)
    assert np.array_equal(x_bins, expected)
    assert np.array_equal(y_bins, expected)

    # test with larger, random data
    position = np.random.uniform(0, 2, (2, 10))
    x_edges = np.arange(0, 2.2, 0.2)
    y_edges = np.arange(0, 2.2, 0.2)
    x_bins, y_bins = compute_spatial_bin_assignment(position, x_edges, y_edges)
    assert isinstance(x_bins, np.ndarray)
    assert isinstance(y_bins, np.ndarray)
    assert position[0].shape == x_bins.shape
    assert position[1].shape == y_bins.shape

    # test warnings
    with warns(UserWarning):
        _ = compute_spatial_bin_assignment(np.array([-1, 1, 2, 3]), np.array([0, 2, 4]))
    with warns(UserWarning):
        _ = compute_spatial_bin_assignment(np.array([[-1, 1, 3, 4], [0, 1, 2, 3]]),
                                           np.array([0, 2, 4]), np.array([0, 2, 4]))

def test_compute_bin_firing():

    bins = [2, 2]
    xbins = [0, 0, 0, 1]
    ybins = [0, 0, 1, 1]

    bin_firing = compute_bin_firing(bins, xbins, ybins)
    assert isinstance(bin_firing, np.ndarray)
    expected = np.array([[2, 1], [0, 1]])
    assert np.array_equal(bin_firing, expected)

def test_compute_bin_time():

    # define a timestamp, with irregular times
    timestamp = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0])
    bin_time = compute_bin_time(timestamp)

    # check dimensions & sum
    assert bin_time.shape[0] == timestamp.shape[0]
    assert np.sum(np.diff(timestamp)) == np.sum(bin_time)

def test_compute_occupancy():

    # Test 1d case
    bins = [3]
    position = np.array([1, 2, 3, 5, 7, 9, 10])
    timestamp = np.linspace(0, 30, len(position))
    occ = compute_occupancy(position, timestamp, bins)
    assert isinstance(occ, np.ndarray)
    assert len(occ) == bins[0]

    # Test 2d case
    bins = [2, 4]
    position = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    timestamp = np.linspace(0, 30, position.shape[1])
    occ = compute_occupancy(position, timestamp, bins)
    assert isinstance(occ, np.ndarray)
    assert occ.shape[0] == BINS[0]
    assert occ.shape[1] == BINS[1]
    # Test flipped binning should get the same total occupancy
    assert np.nansum(occ) == np.nansum(compute_occupancy(position, timestamp, [bins[1], bins[0]]))
