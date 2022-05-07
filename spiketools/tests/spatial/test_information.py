"""Tests for spiketools.spatial.information"""

from spiketools.spatial.information import *

###################################################################################################
###################################################################################################

def test_compute_spatial_information():

    # 1d case: set baseline test values, with no spatial info
    occupancy = np.array([1, 1, 1, 1, 1])
    binned_firing = np.array([1, 1, 1, 1, 1])

    # 1d case - test computation, should be zero information
    spatial_info1 = compute_spatial_information(binned_firing, occupancy)
    assert isinstance(spatial_info1, float)
    assert spatial_info1 == 0.

    # 1d case - check with higher spatially specific firing
    binned_firing_new = np.array([1, 1, 10, 1, 1])
    spatial_info2 = compute_spatial_information(binned_firing_new, occupancy)
    assert isinstance(spatial_info2, float)
    assert spatial_info2 > spatial_info1

    # 1d case - check with proportional occupancy (should be equivalent to previous)
    occupancy_new = occupancy * 2
    spatial_info3 = compute_spatial_information(binned_firing_new, occupancy_new)
    assert isinstance(spatial_info3, float)
    assert np.isclose(spatial_info3, spatial_info2)

    # 2d case: set baseline test values, with no spatial info
    occupancy = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    binned_firing = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])

    # 2d case - test computation
    spatial_info1 = compute_spatial_information(binned_firing, occupancy)
    assert isinstance(spatial_info1, float)
    assert spatial_info1 == 0.

    # 2d case - check with higher spatially specific firing
    binned_firing_new = np.array([[1, 1, 1, 10], [1, 1, 1, 10]])
    spatial_info2 = compute_spatial_information(binned_firing_new, occupancy)
    assert isinstance(spatial_info2, float)
    assert spatial_info2 > spatial_info1

    # 2d case - check with proportional occupancy (should be equivalent to previous)
    occupancy_new = occupancy * 2
    spatial_info3 = compute_spatial_information(binned_firing_new, occupancy_new)
    assert isinstance(spatial_info3, float)
    assert np.isclose(spatial_info3, spatial_info2)
