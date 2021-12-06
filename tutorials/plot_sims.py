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

# import auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

# import all functions from spiketools.sim and helper functions from spiketools.measures and spiketools.plts
from spiketools.sim.dist import sim_spiketrain_binom, sim_spiketrain_poisson
from spiketools.sim.prob import sim_spiketrain_prob
from spiketools.sim.utils import refractory
from spiketools.measures.conversions import convert_train_to_times
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_hist

###################################################################################################
#
# 1. Simulate spikes based on spiking probabilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

###################################################################################################

# simulate spike train of size n_samples based on a probability of spiking per sample
p_spiking = 0.7
n_samples = 100
sim_spiketrain = sim_spiketrain_prob(p_spiking, n_samples)

# convert binary spike train to spike times in milliseconds
spike_times = convert_train_to_times(sim_spiketrain)

# print the first 10 spikes from spike train and spike times
print('Spike train data:', sim_spiketrain[:10])
print('Spike times data:', spike_times[:10])

# plot the spike times
plot_rasters(spike_times)

###################################################################################################

# simulate spike train of size 1000, based on a probability of spiking per sample.
p_spiking = np.random.random(1000)
sim_spiketrain = sim_spiketrain_prob(p_spiking)

###################################################################################################

# convert binary spike train to spike times in milliseconds
spike_times = convert_train_to_times(sim_spiketrain)

# plot the probability distribution of spike times
plot_hist(spike_times, density=1, bins=50,
          xlabel='Spike Times (ms)', ylabel='Probability',
          title='Probability Distribution of Spike Times')

###################################################################################################
#
# 2. Simulate spikes based on different probability distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Simulate spike train from binomial and poisson probability distributions.
#

###################################################################################################

# simulate spike train from binomial probability distribution
p_spiking = 0.7
spikes = sim_spiketrain_binom(p_spiking, n_samples=100)

# convert binary spike train to spike times in milliseconds & plot the spike times
spike_binomial = convert_train_to_times(spikes)
plot_rasters(spike_binomial)

###################################################################################################

# simulate spike train from poisson probability distribution
spikes = sim_spiketrain_poisson(0.16, 100000, 1000, bias=0)

# convert the simulated binary spike train to spike times in milliseconds & plot the spike times
spike_poisson = convert_train_to_times(spikes)
plot_rasters(spike_poisson)

###################################################################################################
#
# 3. Utilities for working with simulated spiking data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Apply a refractory period to a simulated spike train.
#

###################################################################################################

# apply a 0.003 seconds refractory period to the simulated spike train with 1000 Hz sampling rate
spike_ref = refractory(spike_binomial, 0.003, 1000)

# convert binary spike train to spike times in milliseconds & plot the spike times
spike_times = convert_train_to_times(spike_ref)
plt.eventplot(spike_times)

###################################################################################################
