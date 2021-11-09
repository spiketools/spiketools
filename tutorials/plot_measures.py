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

# sphinx_gallery_thumbnail_number = 1

# import auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

# import all functions from spiketools.measures
from spiketools.measures.measures import *
from spiketools.measures.conversions import *

###################################################################################################

# Compute measures of spiking sctivity
# ~~~~~~~~

# First, we estimate spike rate from a vector of spike times, in seconds.
# 

###################################################################################################

# set and visualize a series of spike times
spikes = [0.1, 0.3, 0.9, 1.9, 2.0, 2.1, 2.2, 3.4, 3.5, 3.7, 4.6, 4.8, 4.9, 7]
plt.eventplot(spikes)

###################################################################################################

# compute its spike rate
print('The spike rate is', compute_spike_rate(spikes))

###################################################################################################

# Then, we can compute the inter-spike intervals of that vector of spike times.
#

###################################################################################################

isis = compute_isis(spikes)
print('The inter-spike intervals are', isis)

###################################################################################################

# Next, we can further compute the coefficient of variation of interval-spike intervals we just calculated.
#

###################################################################################################

cv = compute_cv(isis)
print('Coefficient of variation:', cv)

###################################################################################################

# Finally, we can compute the fano factor of a spike train.
#

###################################################################################################

# set a spike train 
spike_train = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]

# compute its fano factor 
print('Fano factor:', compute_fano_factor(spike_train))

###################################################################################################

# Convert spiking data
# ~~~~~~~~

# First, we convert a vector of spike times in millisecond to a binary spike train. 
#

###################################################################################################

# set a vector of spike times in milliseconds
spikes = [1, 4, 5, 6, 9, 16, 17, 20, 25, 28, 40]

# convert spike times to binary spike train
spike_train = create_spike_train(spikes)
print('Spike train:', spike_train)

###################################################################################################

# Similarly, we can also convert binary spike train to spike times in millisecond.
# 

###################################################################################################

# Set a binary spike train
spike_train = [0,0,0,1,0,1,0,0,1,1,1,0,1,1,1,0,0,0,1,1]

# convert binary spike train to spike times in milliseconds
spike_times = convert_train_to_times(spike_train)

print('Spike times:', spike_times)
plt.eventplot(spike_times)

###################################################################################################

# Finally, we can convert a vector of inter-spike intervals in millisecond to spike times in millisecond.
# 

###################################################################################################

# set a vector of inter-spikes interval in milliseconds
isis = [1, 2, 3, 5, 9, 13, 20, 21, 22, 23, 31, 36]
spike_times = convert_isis_to_spikes(isis, offset=0, add_offset=True)

print('Spike times:',spike_times)
plt.eventplot(spike_times)

###################################################################################################
