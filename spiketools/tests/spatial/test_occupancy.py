"""Tests for spiketools.spatial.occupancy"""

from spiketools.spatial.occupancy import *

###################################################################################################
###################################################################################################

def test_compute_spatial_bin_edges():
    pass

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
    assert x_bins.all()==y_bins.all()
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
    assert x_bins.all()==y_bins.all()
    assert position[0].shape == x_bins.shape 
    assert position[1].shape == y_bins.shape

def test_compute_bin_width():
    pass

def test_compute_occupancy():
    pass
