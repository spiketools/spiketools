"""
Simulations
===========

Simulate spiking activity.

This tutorial primarily covers the ``spiketools.sims`` module.
"""

###################################################################################################
# Simulating Place Field Activity
# -----------------------------
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
import matplotlib.pyplot as plt

from spiketools.sim.peak import sim_placefield_peak
from spiketools.sim.placecells import sim_neuron_placefield, sim_trial_placefield

from spiketools.plts.placecells import plot_neuron_placefield

###################################################################################################

# Define simulation parameters
peak_config = {'height': 5.0, 'width': 5.0, 'place_loc': 25, 'n_bins': 50}
placefield_peak = sim_placefield_peak(**peak_config,plot=False)

###################################################################################################

# Plot the place field peak
plot_neuron_placefield(placefield_peak, colormap_name='Greys')

###################################################################################################