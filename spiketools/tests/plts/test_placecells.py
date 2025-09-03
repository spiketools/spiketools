"""Tests for spiketools.plts.neuron"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH
from spiketools.plts.placecells import *
from spiketools.sim.params import *
from spiketools.sim.placecells import *
from spiketools.sim.trials import *
from copy import deepcopy

###################################################################################################
###################################################################################################

@plot_test
def test_plot_neuron_placefield():
    height_vals = [3,4,5,6]
    params = {'n_trials': 2,
            'presence_ratio': 1,
            'base_mean': 0.5,
            'base_std': 0.0,
            'noise_std': 0.2,
            'height_std': 1,
            'width_std': 1,
            'place_loc_std': 1,
            'n_bins': 50,
            'height_mean': 5.0,
            'width_mean': 5.0,
            'place_loc_mean': 25}

    active_params = deepcopy(params)
    param_height = update_vals(active_params, height_vals, upd_height)
    neuron_place_bins=sim_neuron_placefield(param_height)

    vals = np.array([1, 2, 3])
    plot_neuron_placefield(vals, neuron_place_bins, file_path=TEST_PLOTS_PATH, file_name='tplot_neuron_placefield.png')
