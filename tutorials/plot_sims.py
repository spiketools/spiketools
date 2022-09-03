"""
Simulations
===========

Simulate spiking activity.

This tutorial primarily covers the ``spiketools.sims`` module.
"""

###################################################################################################
# Simulating spiking data
# -----------------------
#
# Sections
# ~~~~~~~~
#
# This tutorial contains the following topics:
#
# 1. Simulating spike times, based on specified distributions
# 2. Simulating spike trains, based on spike probability or specified distributions
# 3. Utilities for working with simulated spiking data
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 1

# Import auxiliary libraries
import numpy as np

# Import simulation-related functions
from spiketools.sim import sim_spiketimes, sim_spiketrain
from spiketools.sim.times import sim_spiketimes_poisson
from spiketools.sim.train import sim_spiketrain_binom, sim_spiketrain_poisson, sim_spiketrain_prob
from spiketools.sim.utils import apply_refractory_times, apply_refractory_train

# Import plot functions
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_hist

# Import utilities
from spiketools.measures.conversions import convert_train_to_times

###################################################################################################
# Simulate spike times
# ~~~~~~~~~~~~~~~~~~~~
#
# First we will simulate spike times.
#
# We will start with the :func:`~.sim_spiketimes_poisson`, which simulates
# spike times from a Poisson distribution.
#
# The function takes as input a `rate` and a `duration`.
#

###################################################################################################

# Simulate spike times from a Poisson probability distribution
spike_times_poisson = sim_spiketimes_poisson(rate=5, duration=2)

# Plot the spike times
plot_rasters(spike_times_poisson)

###################################################################################################
# General function for simulation spike times
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the above, we used a specific function for the spike time generation approach.
#
# There is also the more general :func:`~.sim_spiketimes` function,
# with allows for simulating spike times by specifying a method to do so.
#

###################################################################################################

# Simulate a new spike train with the `sim_spiketimes` function
spike_times = sim_spiketimes(7, 2.5, 'poisson')

###################################################################################################
# Utilities for working with simulated spike times
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can apply a refractory period to simulated spike times, using the
# :func:`~.apply_refractory_times` function.
#

###################################################################################################

# Apply a 0.003 seconds refractory period to the simulated spike times
spike_times_ref = apply_refractory_times(spike_times, 0.003)

###################################################################################################
# Simulate spike train
# ~~~~~~~~~~~~~~~~~~~~
#
# Next, let's simulate some spike trains.
#
# Simulating spike trains based on spiking probability
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# First we will simulate spike train based on defining the probability of spiking, with the
# :func:`~.sim_spiketrain_prob` function.
#
# This function takes the following inputs:
# - `p_spiking`, which is the probability (per sample) of spiking
# - `n_samples`, which is the number of samples to simulate
#

###################################################################################################

# Define simulation settings
p_spiking = 0.7
n_samples = 100

# Simulate spike train based on a probability of spiking per sample
spike_train_prob = sim_spiketrain_prob(p_spiking, n_samples)

###################################################################################################
#
# In order to visualize our simulated data, we can convert the spike train to spike times.
#

###################################################################################################

# Convert spike train to spike times in seconds
spike_times_prob = convert_train_to_times(spike_train_prob, fs=1000)

# Print the first 10 spikes from spike train and spike times
print('Spike train data:', spike_train_prob[:10])
print('Spike times data:', spike_times_prob[:10])

###################################################################################################

# Plot the spike times
plot_rasters(spike_times_prob)

###################################################################################################
#
# In the above example, we specified a single spiking probability for all samples.
#
# We can also specify different spiking probability across samples,
# such as in the following example.
#

###################################################################################################

# Simulate spike train of size 1000, based on a probability of spiking per sample
p_spiking = np.random.random(1000)
spike_train_prob_diff = sim_spiketrain_prob(p_spiking)

###################################################################################################

# Convert binary spike train to spike times in seconds
spike_times_prob_diff = convert_train_to_times(spike_train_prob_diff, fs=1000)

# Plot the empirical distribution of spike times probability across the simulated data
plot_hist(spike_times_prob_diff, density=1, bins=50,
          xlabel='Spike Times (seconds)', ylabel='Probability',
          title='Probability Distribution of Spike Times')

###################################################################################################
# Simulate spike trains based on probability distributions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Alternatively, we can simulate spike trains based on different probability distributions.
#
# Options for doing so include:
#
# - :func:`~.sim_spiketrain_binom`, which simulates based on a binomial distribution
# - :func:`~.sim_spiketrain_poisson`, which simulates based on a Poisson distribution
#

###################################################################################################

# Define simulation settings
p_spiking = 0.7

# Simulate spike train from binomial probability distribution
spike_train_binom = sim_spiketrain_binom(p_spiking, n_samples=100)

# Convert binary spike train to spike times in seconds & plot the spike times
spike_times_binomial = convert_train_to_times(spike_train_binom, fs=1000)
plot_rasters(spike_times_binomial)

###################################################################################################

# Simulate spike train from a Poisson probability distribution
spike_train_poisson = sim_spiketrain_poisson(16, 1000, 1000)

# Convert the simulated binary spike train to spike times in seconds & plot the spike times
spike_times_poisson = convert_train_to_times(spike_train_poisson, fs=1000)
plot_rasters(spike_times_poisson)

###################################################################################################
# General function for simulation spike trains
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the above, we used a specific function for the spike train generation approach.
#
# There is also the more general :func:`~.sim_spiketrain` function,
# with allows for simulating spike trains by specifying a method to do so.
#

###################################################################################################

# Use 'sim_spiketrain' to simulate two new spiketrains, with different methods
spike_train1 = sim_spiketrain(0.5, 1000, 'binom')
spike_train2 = sim_spiketrain(12, 1000, 'poisson')

###################################################################################################
# Utilities for working with simulated spike trains
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can apply a refractory period to simulated spike times, using the
# :func:`~.apply_refractory_train` function.
#

###################################################################################################

# Apply a 0.003 seconds refractory period to the simulated spike trains
spike_train_ref1 = apply_refractory_times(spike_train1, 0.003)
spike_train_ref2 = apply_refractory_times(spike_train2, 0.003)

# Plot the simulated data
spike_times1 = convert_train_to_times(spike_train_ref1, fs=1000)
spike_times2 = convert_train_to_times(spike_train_ref2, fs=1000)
plot_rasters([spike_times1, spike_times2])
