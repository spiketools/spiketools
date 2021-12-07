"""
Measures & Conversions
======================

Apply basic measures and conversions to spiking data.

This tutorial primarily covers the ``spiketools.measures`` module.
"""

###################################################################################################
# Applying measures & conversions to spiking data
# -----------------------------------------------
#
# Sections
# ~~~~~~~~
#
# This tutorial contains the following sections:
#
# 1. Compute measures of spiking activity
# 2. Convert spiking data
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 2

# import auxiliary libraries
import numpy as np

# Import measure related functions
from spiketools.measures.measures import compute_spike_rate, compute_isis, compute_cv, compute_fano_factor
from spiketools.measures.conversions import create_spike_train, convert_train_to_times, convert_isis_to_spikes
from spiketools.plts.spikes import plot_isis
from spiketools.sim.prob import sim_spiketrain_prob
from spiketools.plts.trials import plot_rasters

###################################################################################################
#
# Compute measures of spiking activity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we estimate spike rate from a vector of spike times, in milliseconds.
#
# Here, spike time refers to a representation of spiking activity based on listing the times at
# which spikes occur and spike rate measures how fast an individual neuron is firing. Examples of spike
# train and spike times are provided below.
#

###################################################################################################

# generate a binary spike train and its corresponding spike times in milliseconds
p_spiking = np.random.random(100)
spike_train = sim_spiketrain_prob(p_spiking)
spike_times = convert_train_to_times(spike_train)

# print the first 20 spikes in the binary spike train
print(spike_train[:20])

# plot the spike times
plot_rasters(spike_times)

###################################################################################################

# compute the spike rate given a vector of spike times in milliseconds
spike_rate = compute_spike_rate(spike_times)
print('The spike rate is', spike_rate)

###################################################################################################
#
# Then, we can compute the inter-spike intervals, measures of the time intervals between
# successive spikes, of that vector of spike times.
#

###################################################################################################

# compute the interval-spike intervals of a vector of spike times in milliseconds
isis = compute_isis(spike_times)

# plot the inter-spike intervals
plot_isis(isis, bins=None, range=None, density=False, ax=None)

###################################################################################################
#
# Next, we can further compute the coefficient of variation of interval-spike intervals we just calculated.
#

###################################################################################################

# Compute the coefficient of variation
cv = compute_cv(isis)
print('Coefficient of variation:', cv)

###################################################################################################
#
# Finally, we can compute the fano factor, which is a measure of the variability of unit firing, of a spike train.
#

###################################################################################################

# Compute the fano factor of a binary spike train
fano = compute_fano_factor(spike_train)
print('Fano factor: {:1.2f}'.format(fano))

###################################################################################################
#
# Convert spiking data
# ~~~~~~~~~~~~~~~~~~~~
#
# First, we convert a vector of spike times in milliseconds to a binary spike train, which is
# a representation of spiking activity in which spikes are.
#

###################################################################################################

# convert a vector of spike times in milliseconds to a binary spike train
spike_train = create_spike_train(spike_times)

# print the first 20 spikes in the binary spike train
print('Spike train:', spike_train[:20])

###################################################################################################
#
# Similarly, we can also convert binary spike train to spike times in milliseconds.
#

###################################################################################################

# convert a binary spike train to spike times in milliseconds
spike_times = convert_train_to_times(spike_train)

# plot the spike times
plot_rasters(spike_times)

###################################################################################################
#
# Finally, we can convert a vector of inter-spike intervals to spike times.
#

###################################################################################################

# convert a vector of inter-spike intervals in milliseconds to spike times in milliseconds
spike_times = convert_isis_to_spikes(isis, offset=0, add_offset=True)

# plot the spike times
plot_rasters(spike_times)
