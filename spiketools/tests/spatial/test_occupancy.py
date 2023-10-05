"""Tests for spiketools.spatial.occupancy"""

import numpy as np
import pandas as pd

from spiketools.spatial.occupancy import *

###################################################################################################
###################################################################################################

def test_compute_bin_edges():

    ## 1d case
    bins = 3
    position = np.array([0, 1, 2, 3, 4, 5])
    area_range = [1, 4]

    # 1d case with position data
    edges = compute_bin_edges(position, bins)
    assert len(edges) == bins + 1
    assert [edges[0], edges[-1]] == [min(position), max(position)]

    # 1d case with area range
    edges = compute_bin_edges(position, bins, area_range)
    assert np.array_equal(edges, np.array([1.0, 2.0, 3.0, 4.0]))

    # 1d case with no position data (uses area range)
    edges = compute_bin_edges(None, bins, area_range)
    assert np.array_equal(edges, np.array([1.0, 2.0, 3.0, 4.0]))

    ## 2d case
    bins = [2, 4]
    position = np.array([[1., 2., 3., 4., 5.], [0., 1., 2., 3., 4.]])
    area_range = [[1, 4], [1, 4]]

    expected_x = np.array([1.0, 2.5, 4.0])
    expected_y = np.array([1.0, 1.75, 2.5, 3.25, 4.0])

    # 2d case with position data
    x_edges, y_edges = compute_bin_edges(position, bins)
    assert [len(x_edges), len(y_edges)] == [bins[0] + 1, bins[1] + 1]
    assert [x_edges[0], x_edges[-1]] == [min(position[0, :]), max(position[0, :])]
    assert [y_edges[0], y_edges[-1]] == [min(position[1, :]), max(position[1, :])]

    # 2d case with area range
    x_edges, y_edges = compute_bin_edges(position, bins, area_range)
    assert np.array_equal(x_edges, expected_x)
    assert np.array_equal(y_edges, expected_y)

    # 2d case with no position data (uses area range)
    x_edges, y_edges = compute_bin_edges(None, bins, area_range)
    assert np.array_equal(x_edges, expected_x)
    assert np.array_equal(y_edges, expected_y)

    # Test 2d column data
    x_edges_c, y_edges_c = compute_bin_edges(position.T, bins, area_range)
    assert np.array_equal(x_edges_c, expected_x)
    assert np.array_equal(y_edges_c, expected_y)

def test_compute_bin_assignment():

    # test 1d data
    position = np.array([1, 3, 5, 7])
    edges = np.array([0, 2, 4, 6, 8])
    assgns = compute_bin_assignment(position, edges)
    expected1 = np.array([0, 1, 2, 3])
    assert isinstance(assgns, np.ndarray)
    assert np.array_equal(assgns, expected1)

    # test 2d data
    position = np.array([[1, 3, 5, 7], [11, 13, 15, 17]])
    x_edges = np.array([0, 2, 4, 6, 8])
    y_edges = np.array([10, 12, 14, 16, 18])
    x_bins, y_bins = compute_bin_assignment(position, x_edges, y_edges)
    expected2 = np.array([0, 1, 2, 3])
    assert isinstance(x_bins, np.ndarray)
    assert isinstance(y_bins, np.ndarray)
    assert np.array_equal(x_bins, expected1)
    assert np.array_equal(y_bins, expected2)

    # test 2d column data
    x_bins_c, y_bins_c = compute_bin_assignment(position.T, x_edges, y_edges)
    assert np.array_equal(x_bins_c, expected1)
    assert np.array_equal(y_bins_c, expected2)

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
    bin_counts_2d = compute_bin_counts_pos(pos2d, bins2d)
    assert isinstance(bin_counts_2d, np.ndarray)
    assert np.array_equal(bin_counts_2d, np.array([[2, 1, 0], [0, 1, 1]]))

    # test 2d column data
    bin_counts_2dc = compute_bin_counts_pos(pos2d.T, bins2d)
    assert np.array_equal(bin_counts_2dc, np.array([[2, 1, 0], [0, 1, 1]]))

def test_compute_bin_counts_assgn():

    xbins = [0, 0, 0, 1]
    ybins = [0, 0, 1, 2]

    # check 1d case
    bins = 2
    bin_counts = compute_bin_counts_assgn(bins, xbins)
    assert isinstance(bin_counts, np.ndarray)
    assert np.array_equal(bin_counts, np.array([3, 1]))

    # check 2d case
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

    timestamps = np.array([0, 1, 2, 3])

    # 1d case
    bins = 2
    position = np.array([1, 2, 4, 5])
    df = create_position_df(position, timestamps, bins)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == position.shape[-1]
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1, 1]))

    # check speed dropping - drop entries based on min or max speed thresholds
    speed = np.array([1, 1, 0, 2])
    df = create_position_df(position, timestamps, bins, speed=speed, min_speed=0.5)
    assert np.all(df.index == [0, 1, 3]) # check correct index dropped
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1])) # check expected bin output
    df = create_position_df(position, timestamps, bins, speed=speed, max_speed=1.5)
    assert np.all(df.index == [0, 1, 2]) # check correct index dropped
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1])) # check expected bin output

    # check time threshold - drop entries based on min or miax time in bin
    timestamps = np.array([0, 0.5, 1.5, 10])
    df = create_position_df(position, timestamps, bins, min_time=1.0)
    assert np.all(df.index == [1, 2]) # check correct index dropped
    assert np.array_equal(df.xbin.values, np.array([0, 1])) # check expected bin output
    df = create_position_df(position, timestamps, bins, max_time=2.0)
    assert np.all(df.index == [0, 1, 3]) # check correct index dropped
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1])) # check expected bin output

    # 2d case
    bins = [2, 2]
    position = np.array([[1, 2, 4, 5], [5, 4, 2, 1]])
    df = create_position_df(position, timestamps, bins)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == position.shape[-1]
    assert np.array_equal(df.xbin.values, np.array([0, 0, 1, 1]))
    assert np.array_equal(df.ybin.values, np.array([1, 1, 0, 0]))

def test_compute_occupancy_df():

    data_dict = {'time' : np.array([5, 5, 5, 5, 5, 5, 5, 5, 5.]),
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
    data_dict['time'] = np.array([5., 5., 5., 5., 5., 5., 5., 0., 0.])
    bindf = pd.DataFrame(data_dict)
    occ = compute_occupancy_df(bindf, bins, set_nan=True)
    assert np.array_equal(occ, np.array([[10, 5, np.nan], [10, 10, np.nan]]), equal_nan=True)

def test_compute_occupancy():

    # Test 1d case
    bins = 3
    position = np.array([1, 2, 3, 5, 6, 9, 10])
    timestamps = np.linspace(0, 30, len(position))
    occ = compute_occupancy(position, timestamps, bins)
    expected1 = np.array([15, 10, 5])
    assert isinstance(occ, np.ndarray)
    assert occ.shape[0] == bins
    assert np.array_equal(occ, expected1)

    # Test 2d case
    bins = [2, 3]
    position = np.array([[1, 2, 3, 4, 4.5, 5], [6, 7, 8, 8.5, 9.5, 10]])
    timestamps = np.linspace(0, 25, position.shape[1])
    occ = compute_occupancy(position, timestamps, bins)
    expected2 = np.array([[10., 0.], [0., 10.], [0., 5.]])
    assert isinstance(occ, np.ndarray)
    assert np.array_equal(occ.shape, np.array([bins[1], bins[0]]))
    assert np.array_equal(occ, expected2)

    # Test flipped binning should get the same total occupancy
    assert np.nansum(occ) == np.nansum(compute_occupancy(position, timestamps, [bins[1], bins[0]]))

def test_compute_trial_occupancy():

    # Test 1d case
    bins = 3
    position = np.array([1, 2, 3, 5, 6, 9, 10, 2, 4, 6, 7, 8, 9])
    timestamps = np.linspace(0, 60, len(position))
    start_times = [0, 31]
    stop_times = [30, 60]
    trial_occupancy = compute_trial_occupancy(
        position, timestamps, bins, start_times, stop_times)
    expected1 = np.array([[15., 10.,  5.], [10., 5.,  10.]])
    assert isinstance(trial_occupancy, np.ndarray)
    assert trial_occupancy.shape == (len(start_times), bins)
    assert np.array_equal(trial_occupancy, expected1)

    # Test 2d case
    bins = [2, 3]
    position = np.array([[1, 2, 3, 4, 4.5, 5, 2, 2.5, 3.5, 4, 4.5, 5, 5.5],
                         [6, 7, 8, 8.5, 9.5, 10, 6, 6.5, 7, 8, 8.5, 9.5, 10]])
    timestamps = np.linspace(0, 60, position.shape[1])
    start_times = [0, 31]
    stop_times = [30, 60]
    trial_occupancy = compute_trial_occupancy(
        position, timestamps, bins, start_times, stop_times)
    expected2 = np.array([[[10.,  0.], [ 0., 10.], [ 0., 10.]],
                          [[10.,  0.], [ 0., 10.], [ 0.,  5.]]])
    assert isinstance(trial_occupancy, np.ndarray)
    assert trial_occupancy.shape == (len(start_times), bins[1], bins[0])
    assert np.array_equal(trial_occupancy, expected2)
