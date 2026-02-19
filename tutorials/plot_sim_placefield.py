"""
Simulate Place Field
====================

Simulate Place Field Activity.

This tutorial primarily covers the ``spiketools.sims`` module.
"""

###################################################################################################
# Simulating Place Field Activity
# --------------------------------
#
# This tutorial demonstrates how to simulate neural spiking activity that exhibits
# spatial selectivity, specifically place field responses. We'll explore methods for
# generating synthetic place cell data with realistic properties.
#
# Sections
# ~~~~~~~~
#
# This tutorial covers the following topics:
#
# 1. Simulating place field peaks with Gaussian tuning curves
# 2. Adding noise and baseline firing rates
# 3. Simulating place field
# 4. Simulating place field at trial level
# 5. Simulating place cells

###################################################################################################

import numpy as np


from spiketools.sim.peak import sim_placefield_peak
from spiketools.sim.noise import sim_noise, sim_baseline
from spiketools.sim.placefield import sim_placefield
from spiketools.sim.placecells import sim_neuron_placefield, sim_trial_placefield
from spiketools.sim.params import upd_height,update_vals



from spiketools.plts.placecells import plot_neuron_placefield
from spiketools.plts.trials import plot_trial_placefield

###################################################################################################
# Simulate place field peak
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First we will simulate place field peak.
#
# We will start with the :func:`~.sim_placefield_peak`, which simulates
# place field peak from a Gaussian distribution.
#
# The function takes as input a height, width, place location, and number of bins.
#

###################################################################################################

# Define simulation parameters
peak_config = {'height': 5.0, 'width': 5.0, 'place_loc': 25, 'n_bins': 50}
placefield_peak = sim_placefield_peak(**peak_config)

###################################################################################################

# Plot the place field peak
plot_trial_placefield(placefield_peak, ylim=(-1, 10))

###################################################################################################
# Simulate noise
# ~~~~~~~~~~~~~~~
#
# First we will simulate noise.
#
# We will start with the :func:`~.sim_noise`, which simulates
# noise from a Poisson distribution.
#
# The function takes as input a number of bins and noise standard deviation.


###################################################################################################
noise = sim_noise(n_bins=50, noise_std=1)

###################################################################################################

# Plot the noise
plot_trial_placefield(noise, ylim=(-1, 10))

###################################################################################################
# Simulate baseline
# ~~~~~~~~~~~~~~~~~
#
# First we will simulate noise.
#
# We will start with the :func:`~.sim_baseline`, which simulates
# baseline firing rate from a Poisson distribution.
#
# The function takes as input a number of bins and baseline mean and standard deviation.


###################################################################################################
baseline = sim_baseline(n_bins=50, base_mean=2, base_std=0.01)

###################################################################################################

# Plot the noise
plot_trial_placefield(baseline,ylim=(-1, 10))
###################################################################################################
# General function for simulating place field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First we will simulate place field.
#
# We will start with the :func:`~.sim_placefield`, which simulates
# noise from a Poisson distribution.
#
# The function takes as input a height, width, place location, number of bins, noise standard deviation, baseline mean, and baseline standard deviation.


###################################################################################################
placefield_config = {'height': 5.0, 'width': 5.0, 'place_loc': 25, 'n_bins': 50, 'noise_std': .5, 'base_mean': 2, 'base_std': 0.01}
placefield = sim_placefield(**placefield_config)

###################################################################################################

# Plot the noise
plot_trial_placefield(placefield,ylim=(-1, 10))

###################################################################################################
# Simulate trial place field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First we will simulate trial place field.
#
# We will start with the :func:`~.sim_trial_placefield`, which simulates
#
#
# The function takes as input a height mean, height standard deviation, width mean,
# width standard deviation, place location mean,
# place location standard deviation, number of bins,
# noise standard deviation, baseline mean,
# baseline standard deviation, number of trials,
# presence ratio, vary height, vary width,
# and vary place location.

###################################################################################################

trial_placefield_config = {'height_mean': 5.0,
                           'height_std': .5,
                           'width_mean': 5.0,
                           'width_std': 1.0,
                           'place_loc_mean': 25,
                           'place_loc_std': 1,
                           'n_bins': 50,
                           'noise_std': .5,
                           'base_mean': 2,
                           'base_std': 0.01,
                           'n_trials': 64,
                           'presence_ratio': 1,
                           'vary_height': True,
                           'vary_width': False,
                           'vary_place_loc': False}
trial_placefield = sim_trial_placefield(**trial_placefield_config)


###################################################################################################

# Plot the noise
plot_trial_placefield(trial_placefield,average='mean',shade='sem',ylim=(-1, 10), add_traces=True, trace_cmap='Greys')

###################################################################################################
# Simulate place cells with varying height
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First we will simulate place cells with varying height.
#
# We will start with the :func:`~.sim_neuron_placefield`, which simulates
# place cells with varying height.
#
# The function takes as input a height mean, height standard deviation, width mean,
# width standard deviation, place location mean,
# place location standard deviation, number of bins,
# noise standard deviation, baseline mean,
# baseline standard deviation, number of trials,
# presence ratio, vary height, vary width,
# and vary place location.

params = {
    'height_mean': 5.0,
    'height_std': 0.5,
    'width_mean': 5.0,
    'width_std': 1.0,
    'place_loc_mean': 25,
    'place_loc_std': 1,
    'n_bins': 50,
    'noise_std': 0.5,
    'base_mean': 2,
    'base_std': 0.01,
    'n_trials': 64,
    'presence_ratio': 1,
}

# # updater must MODIFY the dict in place
# upd_height = lambda params, val: params.update({'height_mean': val})
height_vals = np.linspace(1, 10, 10)
param_gen = update_vals(params, height_vals, upd_height)
cell_place_bins = sim_neuron_placefield(param_gen)
###################################################################################################

# Plot the place cells with varying height
plot_neuron_placefield(height_vals,cell_place_bins)

###################################################################################################
# Conclusion
# ~~~~~~~~~~
#
# In this tutorial, we covered simulation of place field available in the spiketools module.
#
