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
# 1. Convert spiking data
# 2. Compute measures of spiking activity
# 3. Compute measures of trial spiking activity
# 4. Compute measures of spiking activity pre- and post-event
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
from spiketools.measures.trials import (compute_trial_frs, compute_pre_post_rates,
                                        compute_segment_frs, compute_pre_post_averages,
                                        compute_pre_post_diffs)
from spiketools.plts.spikes import plot_isis
from spiketools.sim.times import sim_spiketimes
from spiketools.plts.trials import plot_rasters
from spiketools.plts.spatial import plot_heatmap

###################################################################################################
# Convert spiking data
# ~~~~~~~~~~~~~~~~~~~~
#
# First, we simulate spike times, in seconds.
#
# Here, spike time refers to a representation of spiking activity based on listing the times at
# which spikes occur and spike rate measures how fast an individual neuron is firing.
# An example of spike spike times is provided below.
#

###################################################################################################

# Generate a spike times at 100Hz for 5 seconds
spike_times = sim_spiketimes(100, 5, 'poisson', refractory=0.001)

###################################################################################################
#
# Now we can convert a vector of spike times in seconds to a binary spike train, a representation
# of spiking activity in which each 1 represents a spike.
# An example spike times and its corresponding spike train are provided below, as well as a raster
# plot of the spike times.
#

###################################################################################################

# Convert a vector of spike times in seconds to a binary spike train using sampling rate of 1000
spike_train = convert_times_to_train(spike_times, fs=1000)

# Print the first 20 spikes in the binary spike train
print('Spike times:', spike_times[spike_times<(20/1000)]) 
print('Corresponding spike train:', spike_train[:20])

# Plot the first second of the spike times
plot_rasters(spike_times[spike_times<1], title='Raster of spike times')

###################################################################################################
#
# Similarly, we can also convert binary spike train to spike times in seconds.
# We can see it has the same raster plot as the original data.
#

###################################################################################################

# Convert a binary spike train with sampling rate of 1000 to spike times in seconds
spike_times = convert_train_to_times(spike_train, fs=1000)

# Plot the first second of the spike times
plot_rasters(spike_times[spike_times<1], title='Raster spike times from spike train')

###################################################################################################
#
# Finally, we can convert from the inter-spike intervals to spike times.
# We can see it has the same raster plot as the original data.
#

###################################################################################################

# Compute the interval-spike intervals of a vector of spike times in seconds
isis = compute_isis(spike_times)
# Convert a vector of inter-spike intervals in seconds to spike times in seconds
spike_times = convert_isis_to_times(isis, offset=0, add_offset=True)

# Plot the first second of the spike times
plot_rasters(spike_times[spike_times<1], title='Raster spike times from ISIS')

###################################################################################################
# Compute measures of spiking activity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we estimate spike rate from a vector of spike times, in spikes/second.
#

###################################################################################################

# Compute the spike rate from spike times
spike_rate = compute_firing_rate(spike_times)
print('The spike rate is  {:.3}'.format(spike_rate), 'Hz')

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
# Next, we can further compute the coefficient of variation of interval-spike
# intervals we just calculated.
#

###################################################################################################

# Compute the coefficient of variation
cv = compute_cv(isis)
print('Coefficient of variation: {:.3}'.format(cv))

###################################################################################################
#
# Finally, we can compute the fano factor, which is a measure of the variability
# of unit firing, of a spike train.
#

###################################################################################################

# Compute the fano factor of a binary spike train
fano = compute_fano_factor(spike_train)
print('Fano factor: {:.3}'.format(fano))

###################################################################################################
# Compute measures of trial spiking activity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we simulate a list of spike times in seconds for 5 different trials.
# For each trial, an event that changes the firing rate is simulated.
# Plot the spiking activity across trials
#

###################################################################################################

# Simulate spiking activity across trials
# For each trial, simulate change in firing rate given an event
# Time before and after event (in seconds)
time_pre = 2
time_post = 4

# Number of trials
n_trials = 5
trial_spikes = [None]*n_trials

# For each trial
for trial_idx in range(n_trials):
    # Generate pre-event spike times: spikes at 5 Hz for 2 seconds (time_pre)
    spikes_pre = sim_spiketimes(5, time_pre, 'poisson', refractory=0.001)

    # Generate post-event spike times: spikes at 10 Hz for 4 seconds (time_post)
    # Add time_pre to the post spikes, since we will stack the pre and the post
    spikes_post = sim_spiketimes(10, time_post, 'poisson', refractory=0.001) + time_pre

    # Stack pre and post, making each trial 6 seconds long
    trial_spikes[trial_idx] = np.append(spikes_pre, spikes_post)

# Plot the spike times across trials
plot_rasters(trial_spikes, title='Spikes per trial')

###################################################################################################
#
# Now we can compute the firing rates for each 2 second bin within each trial.
#

###################################################################################################

# Compute firing rates per trial
# We want the firing rates of 2 second (bin_width) chunks within a trial
bin_width = 2
trial_frs = compute_trial_frs(trial_spikes, bin_width, trange=None, smooth=None)

# Plot trial firing rates (2 second bins)
plot_heatmap(trial_frs, title='Trial FRs, 2s bins')

###################################################################################################
#
# Similarly, we can compute the firing rates for each segment within each trial.
#

###################################################################################################

# Compute firing rates for segments for two example trials
# Note that we can use different segments per trial
segments = np.array([[0, time_pre, time_pre + time_post],
                     [1, time_pre + 1, time_pre + time_post]])
segment_frs = compute_segment_frs(trial_spikes[:2], segments)

# Plot trial firing rates (segments)
plot_heatmap(segment_frs, title='Trial FRs, segments')

###################################################################################################
# Compute measures of spiking activity pre- and post-event
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we calculate the firing rates before and after the event.
#

###################################################################################################

# Compute firing rates pre and post the event for all trials
pre_window = [0, time_pre]
post_window = [time_pre, time_pre + time_post]
frs_pre, frs_post = compute_pre_post_rates(trial_spikes, pre_window, post_window)

###################################################################################################
#
# Now we can compute the average firing rates before and after the event.
# We can also compute the difference between the firing rates before and after the event.
#

###################################################################################################

# Compute the average firing rates
pre_post_avg = compute_pre_post_averages(frs_pre, frs_post, avg_type='mean')
# Print the average firing rates
print('Average FR pre-event: {:.3}'.format(pre_post_avg[0]))
print('Average FR post-event: {:.3}'.format(pre_post_avg[1]))

# Compute the difference between firing rates
pre_post_diffs = compute_pre_post_diffs(frs_pre, frs_post, average=True, avg_type='mean')
# Print the average firing rates
print('Difference between pre- and post-event FR: {:.3}'.format(pre_post_diffs))

###################################################################################################