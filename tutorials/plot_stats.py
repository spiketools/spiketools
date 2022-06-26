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
# 3. Compute f-value from spiking data using ANOVA

###################################################################################################

# sphinx_gallery_thumbnail_number = 3

# Import auxiliary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import statistics-related functions
from spiketools.sim.times import sim_spiketimes
from spiketools.sim.train import sim_spiketrain_binom
from spiketools.stats.shuffle import (shuffle_isis, shuffle_bins, shuffle_poisson,
                                      shuffle_circular)
from spiketools.stats.permutations import compute_surrogate_stats
from spiketools.stats.anova import create_dataframe, fit_anova

# Import spatial analysis functions
from spiketools.spatial.occupancy import (compute_bin_edges, compute_bin_assignment, 
                                          compute_bin_firing)

# Import plot function
from spiketools.plts.trials import plot_rasters
from spiketools.plts.stats import plot_surrogates

# Import measures & utilities
from spiketools.utils.data import restrict_range
from spiketools.measures.measures import compute_firing_rate

###################################################################################################

# Generate spike times in seconds for spikes at 10Hz for 100 seconds
spikes = sim_spiketimes(10, 100, 'poisson', refractory=0.001)
# make sure there are not multiple spikes within the same millisecond
spikes = np.unique((spikes*1000).astype(int)) / 1000

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
shuffled_isis = shuffle_isis(spikes, n_shuffles=10)
shuffled_bins = shuffle_bins(spikes, bin_width_range=[0.5, 7], n_shuffles=10)
shuffled_poisson = shuffle_poisson(spikes, n_shuffles=10)
shuffled_circular = shuffle_circular(spikes, shuffle_min=200, n_shuffles=10)

###################################################################################################

# Plot original spike train
plot_rasters(spikes[:], xlim=[0, 6], title='Non-shuffled', line=None)

###################################################################################################

# Plot different shuffles
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True)

# isis
plot_rasters(shuffled_isis[:, :], xlim=[0, 6], ax=ax1,
             title='Shuffle ISIS n_shuffles = 10', line=None)

# Poisson
plot_rasters(shuffled_poisson[:, :], xlim=[0, 6], ax=ax2,
             title='Shuffle Poisson n_shuffles = 10', line=None)

# shuffled bins
plot_rasters(shuffled_bins[:, :], xlim=[0, 6], ax=ax3,
             title='Shuffle bins n_shuffles = 10', line=None)

# shuffled circular
plot_rasters(shuffled_circular[:, :], xlim=[0, 6], ax=ax4,
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
spikes_pre = sim_spiketimes(5, time_pre, 'poisson', refractory=0.001)

# Generate pre-event spike times: spikes at 10 Hz for 3 seconds (time_post)
time_post = 3
# Add time_pre to the post spikes, since we will stack the pre and the post
spikes_post = sim_spiketimes(10, time_post, 'poisson', refractory=0.001) + time_pre

# Stack pre and post
spikes_pre_post = np.append(spikes_pre, spikes_post)

###################################################################################################

# Get firing rate (spikes/s) post-event (on the final time_post seconds)
fr_post = compute_firing_rate(restrict_range(spikes_pre_post, min_value=time_pre, max_value=None))

# Get firing rate (spikes/s) pre-event (on the initial time_pre seconds)
fr_pre = compute_firing_rate(restrict_range(spikes_pre_post, min_value=None, max_value=time_pre))

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
    fr_post = (compute_firing_rate(restrict_range(shuff_spikes_pre_post[ind, :],
                                                  min_value=time_pre, max_value=None)))
    fr_pre = (compute_firing_rate(restrict_range(shuff_spikes_pre_post[ind, :],
                                                 min_value=None, max_value=time_pre)))
    surr[ind] = fr_post - fr_pre

print(np.min(surr), np.max(surr))

###################################################################################################

# Calculate empirical p-value and z-score of difference in firing rate with respect to surrogates
pval, zscore = compute_surrogate_stats(fr_diff, surr)

print(f'The z-score of the delta firing rate (after - before the event) is {round(zscore, 2)}, \
and the empirical p-value is {round(pval, 2)}')

###################################################################################################

# Plot distribution of surrogates, with calculated delta firing rate & p-value
plot_surrogates(surr, fr_diff, pval)

###################################################################################################
# 3. Compute f-value from spiking data using ANOVA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First generate some spiking data. Noted the spiking data is adopted from previous tutorial of 
# Spatial Analysis.  
# Next, reorganize the computed firing rate (per trial, per bin) into dataframes.  
# Lastly, compute the f-value from the generated spiking data using ANOVA. 
#
# This method can be applied to calculate the f-value from surrogates using ANOVA, which is 
# expected to show relatively similar results to sptatial information method. 
###################################################################################################

# Generate a set of spiking data (same dataset from Spatial Analysis tutorial)
# Set some positional data
x_pos = np.linspace(0, 15, 16)
y_pos = np.array([0, 0, 0.1, 1, 1.5, 1, 1, 2.1, 2, 1, 0, 1, 2, 3, 4, 3.2])
position = np.array([x_pos, y_pos])
# Set number of spatial bins, 3 x-bins and 5 y-bins
bins = [3, 5]
n_bins = bins[0]*bins[1]

# Compute spatial bin edges
x_edges, y_edges = compute_bin_edges(position, bins)

# Set number of trials 
n_trials = 10
bin_firing_all = np.zeros([n_trials,n_bins])

for ind in range(10):
    # Simulate a spike train with chance level
    spike_train = sim_spiketrain_binom(0.5, n_samples=15)

    # Get spike position bins
    spike_bins = np.where(spike_train == 1)[0]
    # Get x and y position bins corresponding to spike positions
    spike_x, spike_y = compute_bin_assignment(position[:, spike_bins], x_edges, y_edges,
                                              include_edge=True)
    # Compute firing rate in each bin
    bin_firing = (compute_bin_firing(bins=bins, xbins=spike_x, ybins=spike_y)).flatten()
    bin_firing_all[ind,:] = bin_firing
    
###################################################################################################

# Organize spiking data into dataframe 
df = create_dataframe(bin_firing_all, ['bin', 'fr'], drop_na=True)

# Compute f_value from spiking data using ANOVA
f_val = fit_anova(df, 'fr ~ C(bin)', 'C(bin)', return_type='f_val', anova_type=2)
print('F-value:', f_val)
