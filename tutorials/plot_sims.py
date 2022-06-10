"""
Simulations
===========

Simulate spiking activity.

This tutorial primarily covers the ``spiketools.sims`` module.
"""

###################################################################################################
#
# Simulation spiking data
# -----------------------
#
# Sections
# ~~~~~~~~
#
# This tutorial contains the following sections:
#
# 1. Simulate spikes based on spiking probabilities
# 2. Simulate spikes based on different probability distributions
# 3. Utilities for working with simulated spiking data
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 1

# Import auxiliary libraries
import numpy as np

# Import simulation-related functions
from spiketools.sim.train import sim_spiketrain_binom, sim_spiketrain_poisson, sim_spiketrain_prob
from spiketools.sim.times import sim_spiketimes
from spiketools.sim.utils import apply_refractory_times

# Import plot functions
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_hist

# Import utitilies
from spiketools.measures.conversions import convert_train_to_times

###################################################################################################
#
# 1. Simulate spikes based on spiking probabilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

###################################################################################################

# Simulate spike train of size n_samples and 1000 Hz sampling rate based on a probability of 
# spiking per sample
p_spiking = 0.7
n_samples = 100
sim_spiketrain = sim_spiketrain_prob(p_spiking, n_samples)

# Convert binary spike train to spike times in seconds
spike_times = convert_train_to_times(sim_spiketrain, fs=1000)

# Print the first 10 spikes from spike train and spike times
print('Spike train data:', sim_spiketrain[:10])
print('Spike times data:', spike_times[:10])

# Plot the spike times
plot_rasters(spike_times)

###################################################################################################

# Simulate spike train of size 1000 and 1000 Hz sampling rate, based on a probability of spiking 
# per sample.
p_spiking = np.random.random(1000)
sim_spiketrain = sim_spiketrain_prob(p_spiking)

###################################################################################################

# Convert binary spike train to spike times in seconds
spike_times = convert_train_to_times(sim_spiketrain, fs=1000)

# Plot the probability distribution of spike times
plot_hist(spike_times, density=1, bins=50,
          xlabel='Spike Times (ms)', ylabel='Probability',
          title='Probability Distribution of Spike Times')

###################################################################################################
#
# 2. Simulate spikes based on different probability distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Simulate spike train from binomial and Poisson probability distributions.
# Simulate spike times from Poisson probability distribution.
#

###################################################################################################

# Simulate spike train from binomial probability distribution
p_spiking = 0.7
spike_train = sim_spiketrain_binom(p_spiking, n_samples=100)

# Convert binary spike train to spike times in seconds & plot the spike times
spike_binomial = convert_train_to_times(spike_train, fs=1000)
plot_rasters(spike_binomial)

###################################################################################################

# Simulate spike train from a Poisson probability distribution
spike_train = sim_spiketrain_poisson(16, 1000, 1000)

# Convert the simulated binary spike train to spike times in seconds & plot the spike times
spike_poisson = convert_train_to_times(spike_train, fs=1000)
plot_rasters(spike_poisson)

###################################################################################################

# Simulate spike times from a Poisson probability distribution
spike_poisson = sim_spiketimes(.16, 100, 'poisson')
# Plot the spike times
plot_rasters(spike_poisson)

###################################################################################################
#
# 3. Utilities for working with simulated spiking data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Apply a refractory period to a simulated spike train.
#

###################################################################################################

# Apply a 0.003 seconds refractory period to the simulated spike train
spike_ref = apply_refractory_times(spike_binomial, 0.003)

# Convert binary spike train to spike times in seconds & plot the spike times
spike_times = convert_train_to_times(spike_ref, fs=1000)
plot_rasters(spike_times)

###################################################################################################
