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

# Import auxiliary libraries
import numpy as np

# Import measure related functions
from spiketools.measures.measures import (compute_firing_rate, compute_isis, compute_cv, 
                                          compute_fano_factor)
from spiketools.measures.conversions import (convert_times_to_train, convert_train_to_times, 
                                             convert_isis_to_times)
from spiketools.plts.spikes import plot_isis
from spiketools.sim.train import sim_spiketrain_prob
from spiketools.plts.trials import plot_rasters

###################################################################################################
#
# Compute measures of spiking activity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we estimate spike rate from a vector of spike times, in seconds.
#
# Here, spike time refers to a representation of spiking activity based on listing the times at
# which spikes occur and spike rate measures how fast an individual neuron is firing. Examples of spike
# train and spike times are provided below.
#

###################################################################################################

# Generate a binary spike train with sampling rate of 1000 and its corresponding spike times in seconds
p_spiking = np.random.random(100)
spike_train = sim_spiketrain_prob(p_spiking)
spike_times = convert_train_to_times(spike_train)

# Print the first 20 spikes in the binary spike train
print(spike_train[:20])

# Plot the spike times
plot_rasters(spike_times)

###################################################################################################

# Compute the spike rate given a vector of spike times in seconds
spike_rate = compute_firing_rate(spike_times)
print('The spike rate is', spike_rate)

###################################################################################################
#
# Then, we can compute the inter-spike intervals, measures of the time intervals between
# successive spikes, of that vector of spike times.
#

###################################################################################################

# Compute the interval-spike intervals of a vector of spike times in seconds
isis = compute_isis(spike_times)

# Plot the inter-spike intervals
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
# First, we convert a vector of spike times in seconds to a binary spike train, which is a
# representation of spiking activity in which spikes are.
#

###################################################################################################

# Convert a vector of spike times in seconds to a binary spike train using sampling rate of 1000 
spike_train = convert_times_to_train(spike_times, fs=1000)

# Print the first 20 spikes in the binary spike train
print('Spike train:', spike_train[:20])

###################################################################################################
#
# Similarly, we can also convert binary spike train to spike times in seconds.
#

###################################################################################################

# Convert a binary spike train with sampling rate of 1000 to spike times in seconds
spike_times = convert_train_to_times(spike_train, fs=1000)

# Plot the spike times
plot_rasters(spike_times)

###################################################################################################
#
# Finally, we can convert a vector of inter-spike intervals to spike times.
#

###################################################################################################

# Convert a vector of inter-spike intervals in seconds to spike times in seconds
spike_times = convert_isis_to_times(isis, offset=0, add_offset=True)

# Plot the spike times
plot_rasters(spike_times)
