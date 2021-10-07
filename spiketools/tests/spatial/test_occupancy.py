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

    # input type check
    assert isinstance(position, np.ndarray)
    assert isinstance(x_edges, np.ndarray)
    assert isinstance(y_edges, np.ndarray)

    # test for simple input (position, x_edges, y_edges)
    bin_assignment_test1 = compute_spatial_bin_assignment(position, x_edges, y_edges)

    # output type check
    assert isinstance(bin_assignment_test1[0], np.ndarray)
    assert isinstance(bin_assignment_test1[1], np.ndarray)

    # output dimension check 
    assert x_bins==y_bins
    assert position[0].shape == bin_assignment_test1[0].shape 
    assert position[1].shape == bin_assignment_test1[1].shape 


    # test 2: test with more random data 
    a = []
    b = []

    for i in range (10):
        np.array(a.append(np.random.uniform(0, 2)))
        np.array(b.append(np.random.uniform(0, 2)))

    position = np.vstack((a,b))
    x_edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
    y_edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])

    # input type check
    assert isinstance(position, np.ndarray)
    assert isinstance(x_edges, np.ndarray)
    assert isinstance(y_edges, np.ndarray)

    # test for input position (randomly generated) and x_edges, y_edges 
    bin_assignment_test2 = compute_spatial_bin_assignment(position, x_edges, y_edges)

    # output type check
    assert isinstance(bin_assignment_test2[0], np.ndarray)
    assert isinstance(bin_assignment_test2[1], np.ndarray)

    # output dimension check 
    assert x_bins==y_bins
    assert position[1].shape == bin_assignment_test2[0].shape 
    assert position[1].shape == bin_assignment_test2[1].shape 

    pass

def test_compute_bin_width():
    pass

def test_compute_occupancy():
    pass
