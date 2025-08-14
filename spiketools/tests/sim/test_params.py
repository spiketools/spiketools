"""Tests for spiketools.sim.params"""

from spiketools.sim.params import *

###################################################################################################
###################################################################################################

def test_upd_height():
    """Test upd_height"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_height(params, 10)
    assert params['height_mean'] == 10


def test_upd_width():
    """Test upd_width"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_width(params, 10)
    assert params['width_mean'] == 10


def test_upd_noise():
    """Test upd_noise"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_noise(params, 2)
    assert params['noise_std'] == 2


def test_upd_placeloc():
    """Test upd_placeloc"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_placeloc(params, 5)
    assert params['place_loc_std'] == 5


def test_upd_skewness():
    """Test upd_skewness"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_skewness(params, 2)
    assert params['skewness_mean'] == 2


def test_upd_presence_ratio():
    """Test upd_presence_ratio"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_presence_ratio(params, 0.5)
    assert params['presence_ratio'] == 0.5


def test_upd_base():
    """Test upd_base"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_base(params, 2)
    assert params['base_mean'] == 2


def test_upd_trials():
    """Test upd_trials"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    upd_trials(params, 20)
    assert params['n_trials'] == 20


def test_upd_npeaks():
    """Test upd_npeaks"""
    params = {'n_height_mean' : [5], 'n_width_mean' : [5], 'n_place_locs_mean' : [5], 'n_place_loc_std' : [5], 'n_height_std' : [5], 'n_width_std' : [5], 
            'n_trials' : 10, 'n_bins' : 10, 'n_peaks' : 1}
    upd_npeaks(params, 2)
    assert params['n_peaks'] == 2
    assert len(params['n_height_mean']) == 2
    assert len(params['n_width_mean']) == 2
    assert len(params['n_place_locs_mean']) == 2
    assert len(params['n_place_loc_std']) == 2
    assert len(params['n_height_std']) == 2
    assert len(params['n_width_std']) == 2


def test_upd_skew_npeaks():
    """Test upd_skew_npeaks"""
    params = {'n_height_mean' : [5], 'n_width_mean' : [5], 'n_place_locs_mean' : [5], 'n_place_loc_std' : [5], 'n_height_std' : [5], 'n_width_std' : [5], 
            'n_skewness_mean' : [5], 'n_skewness_std' : [1], 'n_trials' : 10, 'n_bins' : 10, 'n_peaks' : 1}
    upd_skew_npeaks(params, 3)
    assert params['n_peaks'] == 3
    assert len(params['n_height_mean']) == 3
    assert len(params['n_width_mean']) == 3 
    assert len(params['n_place_locs_mean']) == 3
    assert len(params['n_place_loc_std']) == 3
    assert len(params['n_height_std']) == 3
    assert len(params['n_width_std']) == 3
    assert len(params['n_skewness_mean']) == 3
    assert len(params['n_skewness_std']) == 3


def test_update_vals():
    """Test update_vals"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : 1, 'base_mean' : 1, 'n_trials' : 10}
    # Need to iterate through generator returned by update_vals
    for updated_params in update_vals(params, [10], upd_height):
        pass
    for updated_params in update_vals(params, [10], upd_width):
        pass
    for updated_params in update_vals(params, [2], upd_noise):
        pass
    for updated_params in update_vals(params, [5], upd_placeloc):
        pass
    for updated_params in update_vals(params, [2], upd_skewness):
        pass
    for updated_params in update_vals(params, [0.5], upd_presence_ratio):
        pass
    for updated_params in update_vals(params, [20], upd_trials):
        pass
    assert updated_params['height_mean'] == 10
    assert updated_params['width_mean'] == 10
    assert updated_params['noise_std'] == 2
    assert updated_params['place_loc_std'] == 5
    assert updated_params['skewness_mean'] == 2
    assert updated_params['presence_ratio'] == 0.5
    assert updated_params['n_trials'] == 20


def test_update_paired_vals():
    """Test update_paired_vals"""
    params = {'height_mean' : 5, 'width_mean' : 5, 'noise_std' : 1, 'place_loc_std' : 1, 'skewness_mean' : 1, 'presence_ratio' : .5, 'base_mean' : 1, 'n_trials' : 10}
    for updated_params in update_paired_vals(params, [10], [10], upd_height, upd_width):
        pass
    assert updated_params['height_mean'] == 10
    assert updated_params['width_mean'] == 10
