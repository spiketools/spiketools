"""Tests for spiketools.spatial.occupancy"""

import numpy as np
import pandas as pd

from pytest import warns

from spiketools.spatial.occupancy import *

###################################################################################################
###################################################################################################

def test_compute_nbins():

    # check 1d case
    out1 = compute_nbins([5, 5])
    assert out1 == 25

    # check 2d case
    out2 = compute_nbins([5])
    out3 = compute_nbins(5)
    assert out2 == out3 == 5

def test_compute_bin_edges():

    # check 1d case
    bins = 3
    position = np.array([0, 1, 2, 3, 4, 5])
    edges = compute_bin_edges(position, bins)
    assert len(edges) == bins + 1
    assert edges[0] == min(position)
    assert edges[-1] == max(position)

    # check 2d case - with two inputs of the same size, different number of bins
    bins = [2, 4]
    position = np.array([[1., 2., 3., 4.], [0., 1., 2., 3.]])
    x_edges, y_edges = compute_bin_edges(position, bins)
    assert len(x_edges) == bins[0] + 1
    assert len(y_edges) == bins[1] + 1
    assert x_edges[0] == min(position[0, :])
    assert x_edges[-1] == max(position[0, :])
    assert y_edges[0] == min(position[1, :])
    assert y_edges[-1] == max(position[1, :])
    assert np.all(np.sort(x_edges) == x_edges)
    assert np.all(np.sort(y_edges) == y_edges)

    # checks for two inputs such that one is the other one shuffled
    position = np.array([[1., 2., 3., 4.], [4., 1., 3., 2.]])
    x_edges, y_edges = compute_bin_edges(position, bins)
    assert x_edges[0] == y_edges[0]
    assert x_edges[-1] == y_edges[-1]

    # test for regular input (x) and all zeros input (y)
    position = np.array([[1., 2., 3., 4.], [0., 0., 0., 0.]])
    x_edges, y_edges = compute_bin_edges(position, bins)
    assert np.sum(y_edges == np.linspace(-0.5, 0.5, bins[1] + 1)) == bins[1] + 1

def test_compute_bin_assignment():

    # test with simple data, checking accuracy
    position = np.array([[1, 3, 5, 7], [1, 3, 5, 7]])
    x_edges = np.array([0, 2, 4, 6, 8])
    y_edges = np.array([0, 2, 4, 6, 8])
    x_bins, y_bins = compute_bin_assignment(position, x_edges, y_edges)
    expected = np.array([0, 1, 2, 3])
    assert isinstance(x_bins, np.ndarray)
    assert isinstance(y_bins, np.ndarray)
    assert np.array_equal(x_bins, expected)
    assert np.array_equal(y_bins, expected)

    # test with larger, random data
    position = np.random.uniform(0, 2, (2, 10))
    x_edges = np.arange(0, 2.2, 0.2)
    y_edges = np.arange(0, 2.2, 0.2)
    x_bins, y_bins = compute_bin_assignment(position, x_edges, y_edges)
    assert isinstance(x_bins, np.ndarray)
    assert isinstance(y_bins, np.ndarray)
    assert position[0].shape == x_bins.shape
    assert position[1].shape == y_bins.shape

def test_compute_bin_counts_pos():

    # test 1d case
    pos1d = np.array([0.5, 1.5, 0.5, 2.5, 1.5])
    bins1d = 3
    bin_counts = compute_bin_counts_pos(pos1d, bins1d)
    assert isinstance(bin_counts, np.ndarray)
    assert np.array_equal(bin_counts, np.array([2, 2, 1]))

    # test 2d case
    pos2d = np.array([[0.5, 1.5, 0.5, 2.5, 1.5], [0.5, 1.5, 0.5, 1.5, 0.5]])
    bins2d = [3, 2]
    bin_counts = compute_bin_counts_pos(pos2d, bins2d)
    assert isinstance(bin_counts, np.ndarray)
    assert np.array_equal(bin_counts, np.array([[2, 1, 0], [0, 1, 1]]))

def test_compute_bin_counts_assgn():

    xbins = [0, 0, 0, 1]
    ybins = [0, 0, 1, 2]

    # check 1D case
    bins = 2
    bin_counts = compute_bin_counts_assgn(bins, xbins)
    assert isinstance(bin_counts, np.ndarray)
    assert np.array_equal(bin_counts, np.array([3, 1]))

    # check 2D case
    bins = [2, 3]
    bin_counts = compute_bin_counts_assgn(bins, xbins, ybins)
    assert isinstance(bin_counts, np.ndarray)
    assert np.array_equal(bin_counts.shape, np.array([bins[1], bins[0]]))
    assert np.array_equal(bin_counts, np.array([[2, 0], [1, 0], [0, 1]]))

def test_normalize_bin_counts():

    # Test with full sampling of occupancy
    bin_counts = np.array([[1, 2, 1], [1, 2, 1]])
    occupancy = np.array([[1, 2, 1], [1, 2, 1]])
    normed_counts = normalize_bin_counts(bin_counts, occupancy)
    assert isinstance(normed_counts, np.ndarray)
    assert np.all(normed_counts == 1.)

    # Test with some empty occupancy values (expected nan output)
    bin_counts = np.array([[0, 1, 0], [1, 2, 0]])
    occupancy = np.array([[0, 2, 1], [1, 1, 0]])
    normed_counts = normalize_bin_counts(bin_counts, occupancy)
    assert isinstance(normed_counts, np.ndarray)
    expected = np.array([[np.nan, 0.5, 0.], [1., 2., np.nan]])
    assert np.array_equal(normed_counts, expected, equal_nan=True)

def test_create_position_df():

    timestamps = np.array([5, 5, 5, 5])

    # 1d case
    bins = 2
    position = np.array([1, 2, 4, 5])
    df = create_position_df(position, timestamps, bins)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == position.shape[-1]
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1, 1]))

    # check speed dropping
    speed = np.array([1, 1, 0, 1])
    df = create_position_df(position, timestamps, bins, speed=speed, speed_threshold=0.5)
    assert len(df) == sum(speed)
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1]))

    # 2d case
    bins = [2, 2]
    position = np.array([[1, 2, 4, 5], [5, 4, 2, 1]])
    df = create_position_df(position, timestamps, bins)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == position.shape[-1]
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1, 1]))
    assert np.array_equal(df.ybin.values, np.array([1, 1, 0, 0]))

def test_compute_occupancy_df():

    data_dict = {
        'time' : np.array([5, 5, 5, 5, 5, 5, 5, 5, 5.]),
        'xbin' : np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])}

    # check 1d case
    bins = 3
    bindf = pd.DataFrame(data_dict)
    occ = compute_occupancy_df(bindf, bins)
    assert isinstance(occ, np.ndarray)
    assert np.array_equal(occ, np.array([20, 15, 10]))

    # check 2d case
    data_dict['ybin'] = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1])
    bins = [3, 2]
    bindf = pd.DataFrame(data_dict)
    occ = compute_occupancy_df(bindf, bins)
    assert isinstance(occ, np.ndarray)
    assert np.array_equal(occ.shape, np.array([bins[1], bins[0]]))
    assert np.array_equal(occ, np.array([[10, 5, 5], [10, 10, 5]]))

    # check minimum
    occ = compute_occupancy_df(bindf, bins, minimum=6)
    assert np.array_equal(occ, np.array([[10, 0, 0], [10, 10, 0]]))

    # check normalization
    occ = compute_occupancy_df(bindf, bins, normalize=True)
    assert np.sum(occ) == 1.0

    # check nans
    data_dict['time'] = np.array([5, 5, 5, 5, 5, 5, 5, 0, 0.])
    bindf = pd.DataFrame(data_dict)
    occ = compute_occupancy_df(bindf, bins, set_nan=True)
    assert np.array_equal(occ, np.array([[10, 5, np.nan], [10, 10, np.nan]]), equal_nan=True)

def test_compute_occupancy():

    # Test 1d case
    bins = 3
    position = np.array([1, 2, 3, 5, 6, 9, 10])
    timestamps = np.linspace(0, 30, len(position))
    occ = compute_occupancy(position, timestamps, bins)
    assert isinstance(occ, np.ndarray)
    assert np.array_equal(occ, np.array([15, 10, 5]))
    assert occ.shape[0] == bins

    # Test 2d case
    bins = [2, 3]
    position = np.array([[1, 2, 3, 4, 4.5, 5], [6, 7, 8, 8.5, 9.5, 10]])
    timestamps = np.linspace(0, 25, position.shape[1])
    occ = compute_occupancy(position, timestamps, bins)
    assert isinstance(occ, np.ndarray)
    assert np.array_equal(occ.shape, np.array([bins[1], bins[0]]))
    assert np.array_equal(occ, np.array([[10., 0.], [0., 10.], [0., 5.]]))

    # Test flipped binning should get the same total occupancy
    assert np.nansum(occ) == np.nansum(compute_occupancy(position, timestamps, [bins[1], bins[0]]))
