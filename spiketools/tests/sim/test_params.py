"""Tests for spiketools.sim.params"""

from spiketools.sim.params import *
from spiketools.tests.conftest import t_place_sim_params, t_place_sim_params_npeaks
import copy

###################################################################################################
###################################################################################################

def test_upd_height(t_place_sim_params):
    """Test upd_height"""
    params_copy = copy.deepcopy(t_place_sim_params)
    upd_height(params_copy, 10)
    assert params_copy["height_mean"] == 10

def test_upd_width(t_place_sim_params):
    """Test upd_width"""

    params_copy = copy.deepcopy(t_place_sim_params)
    upd_width(params_copy, 10)
    assert params_copy["width_mean"] == 10


def test_upd_noise(t_place_sim_params):
    """Test upd_noise"""

    params_copy = copy.deepcopy(t_place_sim_params)
    upd_noise(params_copy, 2)
    assert params_copy['noise_std'] == 2


def test_upd_placeloc(t_place_sim_params):
    """Test upd_placeloc"""
    
    params_copy = copy.deepcopy(t_place_sim_params)
    upd_placeloc(params_copy, 5)
    assert params_copy['place_loc_std'] == 5


def test_upd_skewness(t_place_sim_params):
    """Test upd_skewness""" 

    params_copy = copy.deepcopy(t_place_sim_params)
    upd_skewness(params_copy, 2)
    assert params_copy['skewness_mean'] == 2


def test_upd_presence_ratio(t_place_sim_params):
    """Test upd_presence_ratio"""

    params_copy = copy.deepcopy(t_place_sim_params)
    upd_presence_ratio(params_copy, 0.5)
    assert params_copy['presence_ratio'] == 0.5


def test_upd_base(t_place_sim_params):
    """Test upd_base"""

    params_copy = copy.deepcopy(t_place_sim_params)
    upd_base(params_copy, 2)
    assert params_copy['base_mean'] == 2


def test_upd_trials(t_place_sim_params):
    """Test upd_trials"""

    params_copy = copy.deepcopy(t_place_sim_params)
    upd_trials(params_copy, 20)
    assert params_copy['n_trials'] == 20


def test_upd_npeaks(t_place_sim_params_npeaks):
    """Test upd_npeaks"""
    params_copy = copy.deepcopy(t_place_sim_params_npeaks)
    upd_npeaks(params_copy, 2)
    assert params_copy['n_peaks'] == 2
    assert len(params_copy['n_height_mean']) == 2
    assert len(params_copy['n_width_mean']) == 2
    assert len(params_copy['n_place_locs_mean']) == 2
    assert len(params_copy['n_place_loc_std']) == 2
    assert len(params_copy['n_height_std']) == 2
    assert len(params_copy['n_width_std']) == 2


def test_upd_skew_npeaks(t_place_sim_params_npeaks):
    """Test upd_skew_npeaks"""
    params_copy = copy.deepcopy(t_place_sim_params_npeaks)

    upd_skew_npeaks(params_copy, 3)
    assert params_copy['n_peaks'] == 3
    assert len(params_copy['n_height_mean']) == 3
    assert len(params_copy['n_width_mean']) == 3 
    assert len(params_copy['n_place_locs_mean']) == 3
    assert len(params_copy['n_place_loc_std']) == 3
    assert len(params_copy['n_height_std']) == 3
    assert len(params_copy['n_width_std']) == 3
    assert len(params_copy['n_skewness_mean']) == 3
    assert len(params_copy['n_skewness_std']) == 3


def test_update_vals(t_place_sim_params):
    """Test update_vals"""
    params_copy = copy.deepcopy(t_place_sim_params)
    # Need to iterate through generator returned by update_vals
    for updated_params in update_vals(params_copy, [10], upd_height):
        pass
    for updated_params in update_vals(params_copy, [10], upd_width):
        pass
    for updated_params in update_vals(params_copy, [2], upd_noise):
        pass
    for updated_params in update_vals(params_copy, [5], upd_placeloc):
        pass
    for updated_params in update_vals(params_copy, [2], upd_skewness):
        pass
    for updated_params in update_vals(params_copy, [0.5], upd_presence_ratio):
        pass
    for updated_params in update_vals(params_copy, [20], upd_trials):
        pass
    assert updated_params['height_mean'] == 10
    assert updated_params['width_mean'] == 10
    assert updated_params['noise_std'] == 2
    assert updated_params['place_loc_std'] == 5
    assert updated_params['skewness_mean'] == 2
    assert updated_params['presence_ratio'] == 0.5
    assert updated_params['n_trials'] == 20


def test_update_paired_vals(t_place_sim_params):
    """Test update_paired_vals"""
    params_copy = copy.deepcopy(t_place_sim_params)
    for updated_params in update_paired_vals(params_copy, [10], [10], upd_height, upd_width):
        pass
    assert updated_params['height_mean'] == 10
    assert updated_params['width_mean'] == 10
