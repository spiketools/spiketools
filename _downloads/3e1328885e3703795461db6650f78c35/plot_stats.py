"""
Statistical Analyses
====================

Apply statistical analyses to spiking data.

This tutorial primarily covers the ``spiketools.stats`` module.
"""

###################################################################################################
# Applying statistical measures to spiking data
# ---------------------------------------------
#
# This tutorial contains the following sections:
#
# 1. Compute and plot different shuffles of spikes
# 2. Compute empirical p-value and z-score from distribution of surrogates
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 3

# import auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

# Import statistics-related functions
from spiketools.stats.generators import poisson_train
from spiketools.stats.shuffle import (shuffle_isis, shuffle_bins, shuffle_poisson,
                                      shuffle_circular)
from spiketools.stats.permutations import compute_empirical_pvalue, zscore_to_surrogates

# Import plot function
from spiketools.plts.trials import plot_rasters
from spiketools.plts.stats import plot_surrogates

# Import measures & utilities
from spiketools.utils.spikes import restrict_range
from spiketools.measures.measures import compute_spike_rate

###################################################################################################

# Generate spike times for spikes at 10Hz for 100 seconds
poisson_generator = poisson_train(10, 100)
# get spike in seconds
spikes_s = np.array([spike for spike in poisson_generator])
# convert to milliseconds
spikes_ms = np.unique((spikes_s*1000).astype(int))

###################################################################################################
# 1. Compute and plot different shuffles of spikes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we will explore different shuffle approaches to create different shuffled
# surrogates of our original spikes.
#
# The approaches we will try are:
# - `shuffle_isis`: shuffle spike times using permuted inter-spike intervals (isis).
# - `shuffle_bins`: shuffle spikes by creating bins of varying length and then circularly shuffling within those bins.
# - `shuffle_poisson`: shuffle spikes based on a Poisson distribution.
# - `shuffle_circular`: shuffle spikes by circularly shifting the spike train.
#

###################################################################################################

# Shuffle spike ms using the four described methods
shuffled_isis = shuffle_isis(spikes_ms, n_shuffles=10)
shuffled_bins = shuffle_bins(spikes_ms, bin_width_range=[50, 200], n_shuffles=10)
shuffled_poisson = shuffle_poisson(spikes_ms, n_shuffles=10)
shuffled_circular = shuffle_circular(spikes_ms, shuffle_min=200, n_shuffles=10)

###################################################################################################

# Plot original spike train
plot_rasters(spikes_ms[:], xlim=[0, 6000], title='Non-shuffled', line=None)

###################################################################################################

# Plot different shuffles
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True)

# isis
plot_rasters(shuffled_isis[:, :], xlim=[0, 6000], ax=ax1,
             title='Shuffle ISIS n_shuffles = 10', line=None)

# Poisson
plot_rasters(shuffled_poisson[:, :], xlim=[0, 6000], ax=ax2,
             title='Shuffle Poisson n_shuffles = 10', line=None)

# shuffled bins
plot_rasters(shuffled_bins[:, :], xlim=[0, 6000], ax=ax3,
             title='Shuffle bins n_shuffles = 10', line=None)

# shuffled circular
plot_rasters(shuffled_circular[:, :], xlim=[0, 6000], ax=ax4,
             title='Shuffle circular n_shuffles = 10', line=None)

# Add some padding between subplots & make the figure bigger
plt.subplots_adjust(hspace=0.3)
fig.set_size_inches((40/2.54, 20/2.54))

###################################################################################################
# 2. Compute the empirical p-value and z-score from distribution of surrogates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First generate data that simulates a change in spike rate due to an event.
# In this example case, we will be simulating spikes at 1Hz for 3 seconds for the pre-event, and
# spikes at at 5Hz for 3 seconds for the post-event.
# Next, calculate change in firing rate of the post-event with respect to the pre-event
# (delta = post firing rate - pre firing rate).
# To generate our distribution of surrogates, shuffle the data 100 times (using isis), and
# calculate change in firing rate for each of the 100 shuffled spike times.
#
# Compute the empirical p-value and z-score of delta firing rate from the distribution of
# surrogates. Lastly, plot the distribution of surrogates with calculated delta firing rate.
#

###################################################################################################

# Simulate change in firing rate given an event
# Generate pre-event spike times: spikes at 5 Hz for 3 seconds (time_pre)
time_pre = 3
poisson_generator_pre = poisson_train(5, time_pre)
spikes_s_pre = np.array([spike for spike in poisson_generator_pre])

# Generate pre-event spike times: spikes at 10 Hz for 3 seconds (time_post)
time_post = 3
poisson_generator_post = poisson_train(10, time_post)
# add time_pre to the output, since we will stack the pre and the post
spikes_s_post = np.array([spike for spike in poisson_generator_post]) + time_pre

# Stack pre and post
spikes_pre_post = np.append(spikes_s_pre, spikes_s_post)

###################################################################################################

# Get firing rate (spikes/s) post-event (on the final time_post seconds)
fr_post = compute_spike_rate(restrict_range(spikes_pre_post, min_time=time_pre, max_time=None))

# Get firing rate (spikes/s) pre-event (on the initial time_pre seconds)
fr_pre = compute_spike_rate(restrict_range(spikes_pre_post, min_time=None, max_time=time_pre))

# Get firing rate difference between post and pre
# This will be the value we compute the empirical p-value and the z-score for
fr_diff = fr_post - fr_pre

###################################################################################################

# Get shuffled spikes_pre_post (used to calculate surrogates)
n_shuff = 100
shuff_spikes_pre_post = shuffle_isis(spikes_pre_post, n_shuffles=n_shuff)

# Calculate surrogates
# This will be the surrogate distribution used to compute the empirical p-value and the z-score
surr = np.zeros((n_shuff,))
for ind in range(n_shuff):
    fr_post = (compute_spike_rate(restrict_range(shuff_spikes_pre_post[ind, :],
                                                 min_time=time_pre, max_time=None)))
    fr_pre = (compute_spike_rate(restrict_range(shuff_spikes_pre_post[ind, :],
                                                min_time=None, max_time=time_pre)))
    surr[ind] = fr_post - fr_pre

print(np.min(surr), np.max(surr))

###################################################################################################

# Calculate empirical p-value and z-score of difference in firing rate with respect to surrogates
pval = compute_empirical_pvalue(fr_diff, surr)
zscore = zscore_to_surrogates(fr_diff, surr)

print(f'The z-score of the delta firing rate (after - before the event) is {round(zscore, 2)}, \
      and the empirical p-value is {round(pval, 2)}')

###################################################################################################

# Plot distribution of surrogates, with calculated delta firing rate & p-value
plot_surrogates(surr, fr_diff, pval)
