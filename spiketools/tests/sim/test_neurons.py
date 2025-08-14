"""Tests for spiketools.sim.cell"""

from spiketools.sim.neurons import *
from spiketools.sim.params import *

###################################################################################################
###################################################################################################

def test_sim_neuron_placefield():
    """Test sim_neuron_placefield"""
    params = {'height_mean' : 5,'height_std' : 1, 'width_mean' : 5, 'width_std' : 1, 'noise_std' : 1, 'place_loc_mean' : 5, 'place_loc_std' : 1, 'presence_ratio' : .5, 'base_mean' : 1,
              'base_std' : 1, 'n_trials' : 20, 'n_bins' : 100}
    param_gen = update_vals(params, np.linspace(1, 10, 10), upd_height)
    cell_place_bins = sim_neuron_placefield(param_gen)

    assert len(cell_place_bins) == 10
    assert (cell_place_bins[0].shape) == (20, 100)


def test_sim_neuron_skew_placefield():
    """Test sim_neuron_skew_placefield"""
    params = {'height_mean' : 5,'height_std' : 1, 'width_mean' : 5, 'width_std' : 1, 'noise_std' : 1, 'place_loc_mean' : 5, 'place_loc_std' : 1, 'presence_ratio' : .5, 'base_mean' : 1,
              'base_std' : 1, 'skewness_mean' : 1, 'skewness_std' : 1, 'n_trials' : 20, 'n_bins' : 100}
    param_gen = update_vals(params, np.linspace(1, 10, 10), upd_skewness)
    cell_place_bins = sim_neuron_skew_placefield(param_gen)

    assert len(cell_place_bins) == 10
    assert (cell_place_bins[0].shape) == (20, 100)


def test_sim_neuron_multi_placefield():
    """Test sim_neuron_multi_placefield"""
    params = {'n_height_mean' : [5], 'n_width_mean' : [5], 'n_place_locs_mean' : [5], 'n_place_loc_std' : [5], 'n_height_std' : [5], 'n_width_std' : [5], 'base_mean' : 1, 'base_std' : 1,
            'noise_std' : 1, 'presence_ratio' : .5, 'n_trials' : 10, 'n_bins' : 100, 'n_peaks' : 1}
    param_gen = update_vals(params, [1,2,3], upd_npeaks)
    cell_place_bins = sim_neuron_multi_placefield(param_gen)

    assert len(cell_place_bins) == 3
    assert (cell_place_bins[0].shape) == (10, 100)
    assert (cell_place_bins[1].shape) == (10, 100)
    assert (cell_place_bins[2].shape) == (10, 100)


def test_sim_neuron_multi_skew_placefield():
    """Test sim_neuron_multi_skew_placefield"""
    params = {'n_height_mean' : [5], 'n_width_mean' : [5], 'n_place_locs_mean' : [5], 'n_place_loc_std' : [5], 'n_height_std' : [5], 'n_width_std' : [5], 'base_mean' : 1, 'base_std' : 1,
            'n_skewness_mean' : [1], 'n_skewness_std' : [1], 'noise_std' : 1, 'presence_ratio' : .5, 'n_trials' : 10, 'n_bins' : 100, 'n_peaks' : 1}
    param_gen = update_vals(params, [1,2,3], upd_skew_npeaks)
    cell_place_bins = sim_neuron_multi_skew_placefield(param_gen)

    assert len(cell_place_bins) == 3
    assert (cell_place_bins[0].shape) == (10, 100)
    assert (cell_place_bins[1].shape) == (10, 100)
    assert (cell_place_bins[2].shape) == (10, 100)
    