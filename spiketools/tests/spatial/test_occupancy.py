"""Tests for spiketools.spatial.occupancy"""

from spiketools.spatial.occupancy import *
import numpy as np

###################################################################################################
###################################################################################################

def test_compute_spatial_bin_edges():
    # test for regular input (x) and all zeros input (y)
    bins = [4, 4]
    position = np.zeros((2, 4))
    position[0, :] = np.linspace(1, 4, 4)
    x_edges, y_edges = compute_spatial_bin_edges(position, bins)
    # dimension checks
    assert len(x_edges) == bins[0]+1
    assert len(y_edges) == bins[1]+1
    # all zeros case check
    assert np.sum(y_edges == np.linspace(-0.5, 0.5, bins[1]+1)) == bins[1]+1
    # first and last element checks for non all-zero input
    assert x_edges[0] == min(position[0, :])
    assert x_edges[-1] == max(position[0, :])
    # sorting check
    assert bool((np.sort(x_edges) == x_edges).sum())
    assert bool((np.sort(y_edges) == y_edges).sum())
    
    # now check for one input the suffle of the other
    bins = [4, 4]
    position2 = position
    position[1, :] = position[0, :]
    np.random.shuffle(position2[0, :])
    x_edges2, y_edges2 = compute_spatial_bin_edges(position2, bins)
    assert np.sum(x_edges2 == y_edges2) == bins[0]+1

def test_compute_spatial_bin_assignment():
    pass

def test_compute_bin_width():
    pass

def test_compute_occupancy():
    pass
