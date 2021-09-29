"""Tests for spiketools.spatial.occupancy"""

from spiketools.spatial.occupancy import *
import numpy as np

###################################################################################################
###################################################################################################

def test_compute_spatial_bin_edges(position, bins):
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

def test_compute_spatial_bin_assignment():
    pass

def test_compute_bin_width():
    pass

def test_compute_occupancy():
    pass
