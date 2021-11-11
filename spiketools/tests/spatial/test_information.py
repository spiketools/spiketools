"""Tests for spiketools.spatial.information"""

from spiketools.spatial.information import *
from spiketools.spatial.information import _compute_spatial_information

###################################################################################################
###################################################################################################

def test_compute_spatial_information_2d():

    spike_x = [1, 2, 3, 4, 5]
    spike_y = [6, 7, 8, 9, 10]
    bins = [2, 4]

    # Make three different occupancies to compare compute_spatial_information_2d output
    occ_1 = np.array([[1,   1,   1, 1], [1, 250, 250, 1]])
    occ_2 = occ_1 / 10
    occ_3 = np.array([[250, 250, 250, 250], [250, 250, 250, 250]])

    # compute spatial information 2d on three different occupancies
    spatial_info_2d_1 = compute_spatial_information_2d(spike_x, spike_y, bins, occ_1)
    spatial_info_2d_2 = compute_spatial_information_2d(spike_x, spike_y, bins, occ_2)
    spatial_info_2d_3 = compute_spatial_information_2d(spike_x, spike_y, bins, occ_3)

    # dimension check: each of the calculated spatial informations should have a single output
    assert np.array([spatial_info_2d_1, spatial_info_2d_2, spatial_info_2d_3]).shape[0] == 3
    # result check: should be the same for proportional occupancies
    assert np.isclose(spatial_info_2d_1, spatial_info_2d_2)
    # result check: occupancy_3 should return a lower spatial information than occupancy_1
    assert spatial_info_2d_1 > spatial_info_2d_3

def test_compute_spatial_information_1d():

    data = [1, 2, 3, 4, 5]
    bins = [2, 4]
    # make three different occupancies to compare compute_spatial_information_1d output
    occupancy_1 = np.array([1, 250, 250, 1])
    occupancy_2 = occupancy_1/10
    occupancy_3 = np.array([250, 250, 250, 250])

    # compute spatial information 1d on three different occupancies
    spatial_info_1d_1 = compute_spatial_information_1d(data, occupancy_1, bins)
    spatial_info_1d_2 = compute_spatial_information_1d(data, occupancy_2, bins)
    spatial_info_1d_3 = compute_spatial_information_1d(data, occupancy_3, bins)

    # dimension check: each of the calculated spatial informations should have a single output
    assert np.array([spatial_info_1d_1, spatial_info_1d_2, spatial_info_1d_3]).shape[0] == 3
    # result check: should be the same for proportional occupancies
    assert np.isclose(spatial_info_1d_1, spatial_info_1d_2)
    # result check: occupancy_3 should return a lower spatial information than occupancy_1
    assert spatial_info_1d_1 > spatial_info_1d_3

def test_compute_spatial_information():

    # 1d CASE:
    data = [1, 2, 3, 4, 5]
    bins = [2, 4]
    spike_map_1d = np.histogram(data, bins=bins)[0]
    occupancy_1d = np.array([1, 250, 250, 1])
    spatial_info_1d = _compute_spatial_information(spike_map_1d, occupancy_1d)
    assert isinstance(spatial_info_1d, float)

    # 2d CASE:
    spike_x = [1, 2, 3, 4, 5]
    spike_y = [6, 7, 8, 9, 10]
    bins = [2, 4]
    spike_map_2d = np.histogram2d(spike_x, spike_y, bins=bins)[0]
    occupancy_2d = np.array([[1,   1,   1, 1], [1, 250, 250, 1]])
    spatial_info_2d = _compute_spatial_information(spike_map_2d, occupancy_2d)
    assert isinstance(spatial_info_2d, float)
