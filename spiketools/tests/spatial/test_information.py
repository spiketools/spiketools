"""Tests for spiketools.spatial.information"""

from spiketools.tests.tsettings import BINS

from spiketools.spatial.information import *
from spiketools.spatial.information import _compute_spatial_information

###################################################################################################
###################################################################################################

def test_compute_spatial_information_2d():

    spike_x = [1, 2, 3, 4, 5]
    spike_y = [6, 7, 8, 9, 10]

    # Make three different occupancies to compare compute_spatial_information_2d output
    occ_1 = np.array([[1, 1, 1, 1], [1, 250, 250, 1]])
    occ_2 = occ_1 / 10
    occ_3 = np.array([[250, 250, 250, 250], [250, 250, 250, 250]])

    # compute spatial information 2d on three different occupancies
    spatial_info_2d_1 = compute_spatial_information_2d(spike_x, spike_y, BINS, occ_1)
    assert isinstance(spatial_info_2d_1, float)
    spatial_info_2d_2 = compute_spatial_information_2d(spike_x, spike_y, BINS, occ_2)
    assert isinstance(spatial_info_2d_2, float)
    spatial_info_2d_3 = compute_spatial_information_2d(spike_x, spike_y, BINS, occ_3)
    assert isinstance(spatial_info_2d_3, float)

    # result check: should be the same for proportional occupancies
    assert np.isclose(spatial_info_2d_1, spatial_info_2d_2)
    # result check: occ_3 should return a lower spatial information than occ_1
    assert spatial_info_2d_1 > spatial_info_2d_3

def test_compute_spatial_information_1d():

    data = [1, 2, 3, 4, 5]

    # Make three different occupancies to compare compute_spatial_information_1d output
    occ_1 = np.array([1, 250, 250, 1])
    occ_2 = occ_1 / 10
    occ_3 = np.array([250, 250, 250, 250])

    # compute spatial information 1d on three different occupancies
    spatial_info_1d_1 = compute_spatial_information_1d(data, occ_1, BINS)
    assert isinstance(spatial_info_1d_1, float)
    spatial_info_1d_2 = compute_spatial_information_1d(data, occ_2, BINS)
    assert isinstance(spatial_info_1d_2, float)
    spatial_info_1d_3 = compute_spatial_information_1d(data, occ_3, BINS)
    assert isinstance(spatial_info_1d_3, float)

    # result check: should be the same for proportional occupancies
    assert np.isclose(spatial_info_1d_1, spatial_info_1d_2)
    # result check: occ_3 should return a lower spatial information than occ_1
    assert spatial_info_1d_1 > spatial_info_1d_3

def test_compute_spatial_information():

    # 1d CASE:
    data = [1, 2, 3, 4, 5]
    spike_map_1d = np.histogram(data, bins=BINS)[0]
    occ_1d = np.array([1, 250, 250, 1])
    spatial_info_1d = _compute_spatial_information(spike_map_1d, occ_1d)
    assert isinstance(spatial_info_1d, float)

    # 2d CASE:
    spike_x = [1, 2, 3, 4, 5]
    spike_y = [6, 7, 8, 9, 10]
    spike_map_2d = np.histogram2d(spike_x, spike_y, bins=BINS)[0]
    occ_2d = np.array([[1, 1, 1, 1], [1, 250, 250, 1]])
    spatial_info_2d = _compute_spatial_information(spike_map_2d, occ_2d)
    assert isinstance(spatial_info_2d, float)
