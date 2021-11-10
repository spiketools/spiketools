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
# 2. Compute empirical p-value from distribution of spikes
# 3. Compute z-score from distribution of surrogates
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 1

# import auxiliary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import functions from spiketools.stats
from spiketools.stats.generators import poisson_train
from spiketools.stats.shuffle import shuffle_isis, shuffle_bins, shuffle_poisson, shuffle_circular
from spiketools.stats.permutations import compute_empirical_pvalue, zscore_to_surrogates

# import plot_trial_rasters to plot spike trains
from spiketools.plts.trial import plot_trial_rasters

###################################################################################################

# Generate spike times for spikes at 10Hz for 1000 seconds
poisson_generator = poisson_train(10, 1000)
# get spike in seconds
spikes_s = np.array([spike for spike in poisson_generator])
# convert to milliseconds
spikes_ms = spikes_s*1000
spikes_ms = np.unique(spikes_ms.astype(int))

###################################################################################################
# 1. Compute and plot different shuffles of spikes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we will explore different shuffle approaches to create different shuffled surrogates of our original spikes. 
#
# The approaches we will try are:
# - `shuffle_isis`: shuffle spike times using permuted inter-spike intervals (isis).
# - `shuffle_bins`: shuffle spikes by creating bins of varying length and then circularly shuffling within those bins.
# - `shuffle_poisson`: shuffle spikes based on a Poisson distribution.
# - `shuffle_circular`: shuffle spikes by circularly shifting the spike train.
#

###################################################################################################

# Shuffle spike ms using four methods
shuffled_isis = shuffle_isis(spikes_ms, n_shuffles=10)
shuffled_bins = shuffle_bins(spikes_ms, bin_width_range=[50, 200], n_shuffles=10)
shuffled_poisson = shuffle_poisson(spikes_ms, n_shuffles=10)
shuffled_circular = shuffle_circular(spikes_ms, shuffle_min=200, n_shuffles=10)

###################################################################################################

# Plot original spike train
plot_trial_rasters(spikes_ms[:], xlim=[0, 6000], ylim=[0.5, 1.5], title='Non-shuffled')

###################################################################################################

# Plot different shuffles
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True)

# isis
plot_trial_rasters(shuffled_isis[:, 0:80], xlim=[0, 6000], ax=ax1, title='Shuffle ISIS n_shuffles = 10')

# poisson
plot_trial_rasters(shuffled_poisson[:, :], xlim=[0, 6000], ax=ax2, title='Shuffle poisson n_shuffles = 10')

# shuffled bins
plot_trial_rasters(shuffled_bins[:, :], xlim=[0, 6000], ax=ax3, title='Shuffle bins n_shuffles = 10')

# shuffled circular
plot_trial_rasters(shuffled_circular[:, :], xlim=[0, 6000], ax=ax4, title='Shuffle circular n_shuffles = 10')

# Add some padding between subplots
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
# Make figure bigger
f.set_size_inches((40/2.54, 20/2.54))

###################################################################################################
# 2. Compute the empirical p-value from distribution of surrogates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute the empirical p-value of 100 from the distribution of surrogates (spikes_ms).
#

###################################################################################################

# Compute the empirical p-value of 100
compute_empirical_pvalue(100, spikes_ms)

###################################################################################################
# 3. Compute z-score from distribution of surrogates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Z-score number 100 relative to a distribution of surrogates (spikes_ms).
#

###################################################################################################

# Z-score 100 relative to a distribution of surrogates.
zscore_to_surrogates(100, spikes_ms)