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
# 2. CCompute t-test t- and p-values to test for an event-related firing rate change
# 3. Compare 1st and last half of trials
# 3. Run 2-way ANOVA on multiple trials data
# 4. Compute empirical p-value and z-score from distribution of surrogates
# 5. Compute f-value from spiking data using 1-way ANOVA

###################################################################################################

# sphinx_gallery_thumbnail_number = 3

# Import auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

# Import statistics-related functions
from spiketools.stats.shuffle import (shuffle_isis, shuffle_bins, shuffle_poisson,
                                      shuffle_circular)
from spiketools.stats.permutations import compute_surrogate_stats
from spiketools.stats.anova import create_dataframe, create_dataframe_bins, fit_anova
from spiketools.stats.trials import (compute_pre_post_ttest, compare_pre_post_activity, 
                                     compare_trial_frs)

# Import other spiketools functions
from spiketools.sim.times import sim_spiketimes
from spiketools.sim.train import sim_spiketrain_binom
from spiketools.measures.trials import compute_pre_post_rates
from spiketools.spatial.occupancy import (compute_bin_edges, compute_bin_assignment,
                                          compute_bin_counts_assgn)

# Import plot function
from spiketools.plts.trials import plot_rasters
from spiketools.plts.stats import plot_surrogates

# Import measures & utilities
from spiketools.utils.extract import get_range
from spiketools.measures.measures import compute_firing_rate

###################################################################################################

# Generate spike times in seconds for spikes at 10Hz for 100 seconds
# Use the refractory argument to prevent multiple spikes within the same millisecond
spikes = sim_spiketimes(10, 100, 'poisson', refractory=0.001)

###################################################################################################
# Compute and plot different shuffles of spikes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we will explore different shuffle approaches to create different shuffled
# surrogates of our original spikes.
#
# The approaches we will try are:
# - `shuffle_isis`: shuffle spike times using permuted inter-spike intervals (isis)
# - `shuffle_bins`: shuffle spikes by circularly shuffling bins of varying length
# - `shuffle_poisson`: shuffle spikes based on a Poisson distribution
# - `shuffle_circular`: shuffle spikes by circularly shifting the spike train
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
# Compute t-test t- and p-values to test for an event-related firing rate change
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First generate data that simulates a change in spike rate due to an event across 10 trials.
# In this example case, for each trial, we will be simulating spikes at 5Hz for 3 seconds for the 
# pre-event, and spikes at at 10Hz for 3 seconds for the post-event.
#

###################################################################################################

# Simulate spiking activity across trials
# For each trial, simulate change in firing rate given an event
# Time before and after event (in seconds)
time_pre = 3
time_post = 3

# Number of trials
n_trials = 10
trial_spikes = [None]*n_trials

# For each trial
for trial_idx in range(n_trials):
    # Generate pre-event spike times: spikes at 5 Hz for 3 seconds (time_pre)
    spikes_pre = sim_spiketimes(5, time_pre, 'poisson', refractory=0.001)

    # Generate post-event spike times: spikes at 10 Hz for 3 seconds (time_post)
    # Add time_pre to the post spikes, since we will stack the pre and the post
    spikes_post = sim_spiketimes(10, time_post, 'poisson', refractory=0.001) + time_pre

    # Stack pre and post, making each trial 6 seconds long
    trial_spikes[trial_idx] = np.append(spikes_pre, spikes_post)

# Plot the spike times across trials
plot_rasters(trial_spikes, vline=3, title='Spikes per trial')

###################################################################################################
# 
# Next, calculate firing rates of the post-event with respect to the pre-event per trial.
# We will also compute the t-test between the pre- and post-event firing rates across trials.
# 

###################################################################################################

# Compute firing rates (Hz) pre- and post-event for all trials
pre_window = [0, time_pre]
post_window = [time_pre, time_pre + time_post]

frs_pre, frs_post = compute_pre_post_rates(trial_spikes, pre_window, post_window)

###################################################################################################
# 
# Compute the t-test between the pre- and post-event firing rates across trials.
#

###################################################################################################

# Calculate empirical p-value and t-value between the firing rates using a t-test
tval_t, pval_t = compute_pre_post_ttest(frs_pre, frs_post)

print('The t-test value of comparing FRs pre and post the event is {:.3}'.format(tval_t), ', \
 and the t-test p-value is {:.3e}'.format(pval_t))

###################################################################################################
# 
# Next, calculate firing rates of the post-event with respect to the pre-event per trial.
# This is also another way of running the t-test between the pre- and post-event firing rates 
# across trials.
# 

###################################################################################################

# Calculate p-value and t-value between the firing rates using a t-test
# Also calculate the average firing rates pre- and post-event
avg_pre, avg_post, tval_t, pval_t = compare_pre_post_activity(trial_spikes, pre_window, 
                                                              post_window, avg_type='mean')

# Print the average firing rates
print('Average FR pre-event: {:.3}'.format(avg_pre))
print('Average FR post-event: {:.3}'.format(avg_post))

print('The t-test value of comparing FRs pre and post the event is {:.3}'.format(tval_t), '\
 and the t-test p-value is {:.3e}'.format(pval_t))


###################################################################################################
# Compare 1st and last half of trials
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In real data, time or trial index may influence the outcome.
# To check for that, we compare each bin (pre and post event) in the 1st 5 trials to the 
# corresponding bin (pre and post event, respectively) in the last 5 trials using a t-test.
#

###################################################################################################

# Put all trials data into 2-d array
# The data is 10 (n_trials) by 2 (pre and post event bins)
all_trials_data = np.array([frs_pre, frs_post]).T
# For each bin (pre and post event bins), compare 1st half of trials to 2nd half of trials
pre_between_halfs, post_between_halfs = compare_trial_frs(all_trials_data[:5], all_trials_data[5:])

# Print out the results
print('The pre bin, compared between 1st and 2nd half of the trials has a t-value of \
{:.3}'.format(pre_between_halfs[0]), ' and a p-value of {:.3e}'.format(pre_between_halfs[1]))
print('The post bin, compared between 1st and 2nd half of the trials has a t-value of \
{:.3}'.format(post_between_halfs[0]), ' and a p-value of {:.3e}'.format(post_between_halfs[1]))

###################################################################################################
# Run 2-way ANOVA on multiple trials data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Another way of testing for both an event-related and for a trial index main effects on firing 
# rate is to run a 2-way ANOVA.
# To do that, we first organize the firing rates and the factors in a pandas DataFrame.
#

###################################################################################################

# Put the firing rates int a single array
pre_post_frs = np.concatenate([frs_pre, frs_post])
# Create a boolean array to differentiate between pre- and -post event
is_post_event = np.concatenate([np.zeros((len(frs_pre),), dtype=int), np.ones((len(frs_post),), 
                                                                              dtype=int)])
# Create array with trial number information
trial_idx = np.concatenate([np.arange(0, len(frs_pre)), np.arange(0, len(frs_pre))])

# Put it all together into a dictionary
data = {'fr': pre_post_frs, 'is_post_event': is_post_event, 'trial_idx': trial_idx}

# Create the dataframe
df_pre_post = create_dataframe(data, columns=None, drop_na=True, types=None)

###################################################################################################
# 
# Run ANOVA to see the effect of the event on firing rates and the effect of trial index on firing 
# rate. 
# In this example, we did not evaluate the effect of the interaction.
#

###################################################################################################

# Run 2-way ANOVA without interaction on our dataframe
anova_pre_post = fit_anova(df_pre_post, 'fr ~ C(is_post_event)+C(trial_idx)', 
                           return_type='results', anova_type=2)

# Print out results
print('F-value, pre vs post: {:.3}'.format(anova_pre_post['F']['C(is_post_event)']))
print('P-value, pre vs post: {:.3e}'.format(anova_pre_post['PR(>F)']['C(is_post_event)']))
print('F-value, trial index: {:.3}'.format(anova_pre_post['F']['C(trial_idx)']))
print('P-value, trial index: {:.3e}'.format(anova_pre_post['PR(>F)']['C(trial_idx)']))

###################################################################################################
# Compute the empirical p-value and z-score from distribution of surrogates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First we calculate the delta firing rate for an example trial (first trial). Here delta is the 
# difference between pre- and post-even rates (delta = post firing rate - pre firing rate).
# 

###################################################################################################

# Get firing rate difference between post and pre
# This will be the value we compute the empirical p-value and the z-score for
fr_diff = frs_post[0] - frs_pre[0]

###################################################################################################
# 
# To compare the delta firing rate, we need to generate our distribution of surrogates.
# Thus, we will shuffle the data from the first trial (as an example) 100 times (using isis).
# For each of the shuffles, we will calculate change in firing rate.
#

###################################################################################################

# Get shuffled spikes_pre_post (used to calculate surrogates)
n_shuff = 100
shuff_spikes = shuffle_isis(trial_spikes[0], n_shuffles=n_shuff)

# Calculate surrogates
# This will be the surrogate distribution used to compute the empirical p-value and the z-score
shuff_frs_pre, shuff_frs_post = compute_pre_post_rates(shuff_spikes, pre_window, post_window)
shuff_fr_diff = shuff_frs_post - shuff_frs_pre

print('Minimum delta FR across surrogates: {:.3}'.format(np.min(shuff_fr_diff)))
print('Maximum delta FR across surrogates: {:.3}'.format(np.max(shuff_fr_diff)))

###################################################################################################
# 
# Compute the empirical p-value and z-score of delta firing rate from the distribution of
# surrogates. Lastly, plot the distribution of surrogates with calculated delta firing rate.
#

###################################################################################################

# Calculate empirical p-value and z-score of difference in firing rate with respect to surrogates
surr_pval, surr_zscore = compute_surrogate_stats(fr_diff, shuff_fr_diff)

print('The z-score of the delta FR (after - before the event) is {:.3}'.format(surr_zscore), \
'and the empirical p-value is {:.3e}'.format(surr_pval))

###################################################################################################

# Plot distribution of surrogates, with calculated delta firing rate & p-value
plot_surrogates(shuff_fr_diff, fr_diff, surr_pval)

###################################################################################################
# Compute f-value from spiking data using 1-way ANOVA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First generate some spiking data. The simulated spike train has 8 seconds of data across trials
# and has a firing rate between 2-10 Hz, assumed 1k Hz sampling rate.
# Next, reorganize the computed firing rate (per trial, per bin) into dataframes.
# Lastly, compute the f-value from the generated spiking data using ANOVA.
#
# This method can also be applied to calculate the f-value from surrogates using ANOVA.

###################################################################################################

# Generate a set of spiking data (same dataset from Spatial Analysis tutorial)
# Set some positional data
x_pos = np.linspace(0, 15, 8000)
y_pos = np.linspace(0, 5, 8000)
position = np.array([x_pos, y_pos])
# Set number of spatial bins, 3 x-bins and 5 y-bins
bins = [3, 5]
n_bins = bins[0]*bins[1]

# Compute spatial bin edges
x_edges, y_edges = compute_bin_edges(position, bins)

# Set number of trials
n_trials = 10
bin_firing_all = np.zeros([n_trials,n_bins])

for ind in range(n_trials):
    # Simulate a spike train with a sampling rate of 1k Hz
    spike_train = sim_spiketrain_binom(0.005, n_samples=8000)

    # Get spike position bins
    spike_bins = np.where(spike_train == 1)[0]
    # Get x and y position bins corresponding to spike positions
    spike_x, spike_y = compute_bin_assignment(position[:, spike_bins], x_edges, y_edges,
                                              include_edge=True)
    # Compute firing rate in each bin
    bin_firing = (compute_bin_counts_assgn(bins=bins, xbins=spike_x, ybins=spike_y)).flatten()
    bin_firing_all[ind,:] = bin_firing

###################################################################################################

# Organize spiking data into dataframe
df = create_dataframe_bins(bin_firing_all, ['bin', 'fr'], drop_na=True)

# Compute f_value from spiking data using ANOVA
f_val = fit_anova(df, 'fr ~ C(bin)', feature='C(bin)', return_type='f_val', anova_type=1)
print('F-value: {:.3}'.format(f_val))

###################################################################################################