"""
Simulations
===========

Simulate spiking activity.

This tutorial primarily covers the ``spiketools.sims`` module.
"""

###################################################################################################
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

###################################################################################################

# sphinx_gallery_thumbnail_number = 1

# import auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# import all functions from spiketools.sim and a supporting function from spiktool.measures
from spiketools.sim.dist import sim_spiketrain_binom, sim_spiketrain_poisson
from spiketools.sim.prob import sim_spiketrain_prob
from spiketools.sim.utils import refractory
from spiketools.measures.conversions import convert_train_to_times

###################################################################################################

# 1. Simulate spikes based on spiking probabilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

###################################################################################################

# Simulate spike train of size n_samples based on a probability of spiking per sample
p_spiking = 0.7
n_samples = 30
sim_spiketrain = sim_spiketrain_prob(p_spiking, n_samples)

###################################################################################################

# Convert binary spike train to spike times in milliseconds & plot the spike times
spike_times = convert_train_to_times(sim_spiketrain)
plt.eventplot(spike_times)

###################################################################################################

# Additionally, we can simulate spike train of size n_samples, based on a probability of spiking per sample.
p_spiking = np.array([random.random() for i in range(1000)])
sim_spiketrain = sim_spiketrain_prob(p_spiking)

###################################################################################################

# Convert binary spike train to spike times in milliseconds
spike_times = convert_train_to_times(sim_spiketrain)

# Plot the probability distribution of spike times
plt.title('Probability Distribution of Spike Times')
plt.xlabel('Spike Times (ms)')
plt.ylabel('Probability')
plt.hist(spike_times, density=1, bins=50)
plt.show()

###################################################################################################

# 2. Simulate spikes based on different probability distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Simulate spike train from binomial and poisson probability distributions.
#

###################################################################################################

# Simulate spike train from binomial probability distribution
p_spiking = 0.7
spikes = sim_spiketrain_binom(p_spiking, n_samples=50)

# Convert binary spike train to spike times in milliseconds & plot the spike times
spike_times = convert_train_to_times(spikes)
plt.eventplot(spike_times)

###################################################################################################

# Simulate spike train from poisson probability distribution
spikes = sim_spiketrain_poisson(0.4, 50, 1000, bias=0)

###################################################################################################

# 3. Utilities for working with simulated spiking data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Apply a refractory period to a simulated spike train.
#

###################################################################################################

# Use the spike train simulated eariler from a binomial probability distribution
p_spiking = 0.7
spikes = sim_spiketrain_binom(p_spiking, n_samples=50)

# Convert binary spike train to spike times in milliseconds & plot the spike times
spike_before = convert_train_to_times(spikes)
plt.eventplot(spike_before)

###################################################################################################
# Apply a 0.3 seconds refractory period to a simulated spike train with 1000 Hz sampling rate
spike_ref = refractory(spikes, 0.3, 1000)

# Convert binary spike train to spike times in milliseconds & plot the spike times
spike_after = convert_train_to_times(spike_ref)
plt.eventplot(spike_after)

###################################################################################################
