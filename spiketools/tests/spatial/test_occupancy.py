"""Tests for spiketools.spatial.occupancy"""

from spiketools.spatial.occupancy import *
import numpy as np

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
    assert np.sum(x_edges == y_edges) == bins[0]+1
    
    # test for regular input (x) and all zeros input (y)
    position = np.array([[1., 2., 3., 4.], [0., 0., 0., 0.]])
    x_edges, y_edges = compute_spatial_bin_edges(position, bins)
    # all zeros case check
    assert np.sum(y_edges == np.linspace(-0.5, 0.5, bins[1]+1)) == bins[1]+1

def test_compute_spatial_bin_assignment():
    pass

def test_compute_bin_width():
    pass

def test_compute_occupancy():
    pass
