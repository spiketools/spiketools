"""Tests for spiketools.spatial.distance"""

import numpy as np

from spiketools.spatial.distance import *

###################################################################################################
###################################################################################################

def test_compute_distance():

    # 1d
    out1d0 = compute_distance([1], [1])
    assert isinstance(out1d0, float)
    assert np.isclose(out1d0, 0.0)

    out1d1 = compute_distance([1], [2])
    assert isinstance(out1d1, float)
    assert np.isclose(out1d1, 1.0)

    # 2d
    out2d0 = compute_distance([1, 1], [1, 1])
    assert isinstance(out2d0, float)
    assert np.isclose(out2d0, 0.0)

    out2d1 = compute_distance([1, 1], [2, 1])
    assert isinstance(out2d1, float)
    assert np.isclose(out2d1, 1.0)

    out2d2 = compute_distance([1, 1], [2, 2])
    assert np.isclose(out2d2, np.sqrt(2))

    out2d3 = compute_distance([1, 1], [-1, 1])
    assert np.isclose(out2d3, 2)

def test_compute_distances():

    # 1d
    pos1d = np.array([0, 0, 1, 1, 2])
    out1d = compute_distances(pos1d)
    assert isinstance(out1d, np.ndarray)
    assert len(out1d) == pos1d.shape[-1] - 1
    assert np.allclose(out1d, np.array([0, 1, 0, 1]))

    # 2d
    pos2d = np.array([[0, 0, 1, 1, 2],
                      [0, 0, 0, 1, 2]])
    out2d = compute_distances(pos2d)
    assert isinstance(out2d, np.ndarray)
    assert len(out2d) == pos2d.shape[-1] - 1
    assert np.allclose(out2d, np.array([0, 1, 1, np.sqrt(2)]))

def test_compute_cumulative_distances():

    # 1d
    pos1d = np.array([0, 0, 1, 1, 2])
    out1d = compute_cumulative_distances(pos1d, align_output=False)
    assert isinstance(out1d, np.ndarray)
    assert len(out1d) == pos1d.shape[-1] - 1
    assert np.allclose(out1d, np.array([0, 1, 1, 2]))
    # Test with alignment
    assert np.allclose(compute_cumulative_distances(pos1d), np.array([0, 0, 1, 1, 2]))

    # 2d
    pos2d = np.array([[0, 0, 1, 1, 2],
                      [0, 0, 0, 1, 2]])
    out2d = compute_cumulative_distances(pos2d, align_output=False)
    assert isinstance(out2d, np.ndarray)
    assert len(out2d) == pos2d.shape[-1] - 1
    assert np.allclose(out2d, np.array([0, 1, 1 + 1, 1 + 1 + np.sqrt(2)]))
    # Test with alignment
    assert np.allclose(compute_cumulative_distances(pos2d),
                       np.array([0, 0, 1, 1 + 1, 1 + 1 + np.sqrt(2)]))

def test_compute_distances_to_location():

    # 1d
    pos1d = np.array([0, 0, 1, 1, 2])
    loc1d = 1

    out1d = compute_distances_to_location(pos1d, loc1d)
    assert isinstance(out1d, np.ndarray)
    assert len(out1d) == pos1d.shape[-1]
    assert np.allclose(out1d, np.array([1, 1, 0, 0, 1]))

    # 2d
    pos2d = np.array([[0, 0, 1, 1, 2],
                      [0, 0, 0, 1, 2]])
    loc2d = [1, 0]
    out2d = compute_distances_to_location(pos2d, loc2d)
    assert isinstance(out2d, np.ndarray)
    assert len(out2d) == pos2d.shape[-1]
    assert np.allclose(out2d, np.array([1, 1, 0, 1, np.sqrt(5)]))

def test_get_closest_location():

    # 1d
    pos1d = np.array([0, 1, 2])

    assert get_closest_position(pos1d, 1) == 1
    assert get_closest_position(pos1d, 2) == 2
    assert get_closest_position(pos1d, 1.75) == 2
    assert get_closest_position(pos1d, 1.5, threshold=0.25) == -1

    # 2d
    pos2d = np.array([[0, 1, 2],
                      [0, 0, 2]])
    assert get_closest_position(pos2d, [1, 0]) == 1
    assert get_closest_position(pos2d, [2, 2]) == 2
    assert get_closest_position(pos2d, [1, 1]) == 1
    assert get_closest_position(pos2d, [1, 1], threshold=0.25) == -1
