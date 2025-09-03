"""Tests for spiketools.sim.place_cells"""

from spiketools.sim.place_cells import *
from spiketools.sim.params import *
from spiketools.tests.conftest import t_place_sim_params, t_place_sim_skew_params, t_place_sim_params_npeaks, t_place_sim_params_npeaks_skew
import copy

###################################################################################################
###################################################################################################

def test_sim_neuron_placefield(t_place_sim_params):
    """Test sim_neuron_placefield"""
    params_copy = copy.deepcopy(t_place_sim_params)
    param_gen = update_vals(params_copy, np.linspace(1, 10, 10), upd_height)
    cell_place_bins = sim_neuron_placefield(param_gen)

    assert len(cell_place_bins) == 10
    assert (cell_place_bins[0].shape) == (20, 100)


def test_sim_neuron_skew_placefield(t_place_sim_skew_params):
    """Test sim_neuron_skew_placefield"""
    params_copy = copy.deepcopy(t_place_sim_skew_params)
    param_gen = update_vals(params_copy, np.linspace(1, 10, 10), upd_skewness)
    cell_place_bins = sim_neuron_skew_placefield(param_gen)

    assert len(cell_place_bins) == 10
    assert (cell_place_bins[0].shape) == (20, 100)


def test_sim_neuron_multi_placefield(t_place_sim_params_npeaks):
    """Test sim_neuron_multi_placefield"""
    params_copy = copy.deepcopy(t_place_sim_params_npeaks)
    param_gen = update_vals(params_copy, np.linspace(1, 10, 10), upd_npeaks)
    cell_place_bins = sim_neuron_multi_placefield(param_gen)

    assert len(cell_place_bins) == 10
    assert (cell_place_bins[0].shape) == (20, 100)


    param_gen = update_vals(params, [1,2,3], upd_npeaks)
    cell_place_bins = sim_neuron_multi_placefield(param_gen)

    assert len(cell_place_bins) == 3
    assert (cell_place_bins[0].shape) == (10, 100)
    assert (cell_place_bins[1].shape) == (10, 100)
    assert (cell_place_bins[2].shape) == (10, 100)


def test_sim_neuron_multi_skew_placefield(t_place_sim_params_npeaks_skew):
    """Test sim_neuron_multi_skew_placefield"""
    params_copy = copy.deepcopy(t_place_sim_params_npeaks_skew)
    param_gen = update_vals(params_copy, [1,2,3], upd_skew_npeaks)
    cell_place_bins = sim_neuron_multi_skew_placefield(param_gen)

    assert len(cell_place_bins) == 3
    assert (cell_place_bins[0].shape) == (10, 100)
    assert (cell_place_bins[1].shape) == (10, 100)
    assert (cell_place_bins[2].shape) == (10, 100)
    