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
# Sections
# ~~~~~~~~
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

# import all functions from spiketools.stats
from spiketools.stats.generators import *
from spiketools.stats.permutations import *
from spiketools.stats.shuffle import *

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
# ~~~~~~~~
#
# Use spikes_ms to compute shuffle_isis (with 10 shuffles), shuffle_bins (with 10 shuffles), 
# shuffle_poisson (with 10 shuffles), and shuffle_circular (with 10 shuffles).
# Also compute a vector permutation, for comparison.
#

###################################################################################################

# Shuffle spike ms using four methods
shuffled_isis = shuffle_isis(spikes_ms, n_shuffles=10)
shuffled_bins = shuffle_bins(spikes_ms, bin_width_range=[50, 200], n_shuffles=10)
shuffled_poisson = shuffle_poisson(spikes_ms, n_shuffles=10)
shuffled_circular = shuffle_circular(spikes_ms, shuffle_min=200, n_shuffles=10)

# Do a vector permutation, for comparison
perm_spikes = vec_perm(spikes_ms, n_perms=10)

###################################################################################################

# Plot non-shuffled, all different shuffles, and vector permutation
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey = True)
# original
ax1.eventplot(spikes_ms[0:50])
ax1.set_title('Non-shuffled')
ax1.set_xlim([0, 6000])
ax1.set_xlabel('time')

# isis
ax2.eventplot(shuffled_isis[:, 0:80])
ax2.set_title('Shuffle ISIS n_shuffles = 10')
ax2.set_xlim([0, 6000])
ax2.set_xlabel('time')

# poisson
ax3.eventplot(shuffled_poisson[:, :])
ax3.set_title('Shuffle poisson n_shuffles = 10')
ax3.set_xlim([0, 6000])
ax3.set_xlabel('time')

# only plot up to 55th so the effect is visible
ax4.eventplot(perm_spikes[:, :55])
ax4.set_title('Permutated spikes n_permutations = 10')
ax4.set_xlim([0, 6000])
ax4.set_xlabel('time')

# shuffled bins
ax5.eventplot(shuffled_bins[:, :])
ax5.set_title('Shuffle bins n_shuffles = 10')
ax5.set_xlim([0, 6000])
ax5.set_xlabel('time')

# shuffled circular
ax6.eventplot(shuffled_circular[:, :])
ax6.set_title('Shuffle circular n_shuffles = 10')
ax6.set_xlim([0, 6000])
ax6.set_xlabel('time')

# Add some padding between subplots
# f.tight_layout(h_pad = 0.001)
# Make figure bigger
f.set_size_inches((40/2.54, 20/2.54))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

###################################################################################################
# 2. Compute the empirical p-value from distribution of surrogates
# ~~~~~~~~
#
# Compute the empirical p-value of 100 from the distribution of surrogates (spikes_ms).
#

###################################################################################################

# Compute the empirical p-value of 100
compute_empirical_pvalue(100, spikes_ms)

###################################################################################################
# 3. Compute z-score from distribution of surrogates
# ~~~~~~~~
#
# Z-score number 100 relative to a distribution of surrogates (spikes_ms).
#

###################################################################################################

# Z-score 100 relative to a distribution of surrogates.
zscore_to_surrogates(100, spikes_ms)