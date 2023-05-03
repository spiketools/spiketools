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
# This tutorial covers applying statistical measures to spike data.
#
# This tutorial contains the following sections:
#
# 1. Compute and plot different shuffles of spikes
# 2. Compute t-test t- and p-values to test for an event-related firing rate change
# 3. Compare first and last half of trials
# 4. Run a 2-way ANOVA on multiple trials data
# 5. Compute empirical p-value and z-score from distribution of surrogates
# 6. Compute f-value from spiking data using 1-way ANOVA
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 5

# Import auxiliary libraries
import numpy as np

# Import statistics-related functions
from spiketools.stats.shuffle import (shuffle_spikes, shuffle_isis, shuffle_bins,
                                      shuffle_poisson, shuffle_circular)
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
from spiketools.plts.utils import make_axes

###################################################################################################
#
# First we will generate some test data, by simulating some spike times.
#
# To do so, we will simulate some spike times at 10Hz for 100 seconds.
# In doing so, we will use the `refractory` settings to prevent multiple
# spikes occurring within the same millisecond
#

###################################################################################################

# Simulate spike times
spikes = sim_spiketimes(10, 100, 'poisson', refractory=0.001)

###################################################################################################
# Compute and plot different shuffles of spikes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we will explore different shuffle approaches to create different shuffled
# surrogates of our original spikes.
#
# The approaches we will try are:
#
# - :func:`~.shuffle_isis`: shuffle spike times using permuted inter-spike intervals (isis)
# - :func:`~.shuffle_bins`: shuffle spikes by circularly shuffling bins of varying length
# - :func:`~.shuffle_poisson`: shuffle spikes based on a Poisson distribution
# - :func:`~.shuffle_circular`: shuffle spikes by circularly shifting the spike train
#

###################################################################################################

# Shuffle spikes using the four described methods
shuffled_isis = shuffle_isis(spikes, n_shuffles=10)
shuffled_bins = shuffle_bins(spikes, bin_width_range=[0.5, 7], n_shuffles=10)
shuffled_poisson = shuffle_poisson(spikes, n_shuffles=10)
shuffled_circular = shuffle_circular(spikes, shuffle_min=200, n_shuffles=10)

###################################################################################################

# Plot original spike train
plot_rasters(spikes, xlim=[0, 6], title='Non-shuffled', vline=None)

###################################################################################################

# Plot different shuffles
ax1, ax2, ax3, ax4 = make_axes(4, 2, sharey=True, hspace=0.3, figsize=(15, 7))

# Shuffle spikes based on inter-spike intervals
plot_rasters(shuffled_isis, xlim=[0, 6], ax=ax1,
             title='Shuffle ISIS n_shuffles = 10')

# Shuffle spikes Poisson
plot_rasters(shuffled_poisson, xlim=[0, 6], ax=ax2,
             title='Shuffle Poisson n_shuffles = 10')

# Shuffled spikes using a binned circular shuffle
plot_rasters(shuffled_bins, xlim=[0, 6], ax=ax3,
             title='Shuffle bins n_shuffles = 10')

# Circular shuffle
plot_rasters(shuffled_circular, xlim=[0, 6], ax=ax4,
             title='Shuffle circular n_shuffles = 10')

###################################################################################################
#
# In the above we ran different shuffling approaches by calling different shuffling functions.
#
# There is also the more general :func:`~.shuffle_spikes` function, which can be used
# call the different shuffling methods by passing an argument to define which method to use.
#

###################################################################################################

# Shuffle spikes using the general `shuffle_spikes` function
shuffled_spikes = shuffle_spikes(spikes, 'CIRCULAR', shuffle_min=200, n_shuffles=10)

# Plot shuffle spikes
plot_rasters(shuffled_spikes, xlim=[0, 6], title='Shuffled Spikes')

###################################################################################################
# Compute t-test t- and p-values to test for an event-related firing rate change
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First generate data that simulates a change in spike rate due to an event across 10 trials.
#
# In this example case, across trials, we will simulate a change in firing rate
# around an event. Specifically, we will simulate trials with an average firing rate of
# 5 Hz in the 3 second pre-event window, which changes to a firing rate of 10 Hz in
# 3 second post-event window.
#

###################################################################################################

# Set the number of trials to simulate
n_trials = 10

# Define settings for the simulation
time_pre = 3
fr_pre = 5
time_post = 3
fr_post = 10

# Simulate spiking activity across trials
trial_spikes = [None] * n_trials
for trial_idx in range(n_trials):

    # Generate pre- and post-event spike times, offsetting post spikes to happen after pre
    spikes_pre = sim_spiketimes(fr_pre, time_pre, 'poisson', refractory=0.001)
    spikes_post = sim_spiketimes(fr_post, time_post, 'poisson', refractory=0.001) + time_pre

    # Stack pre and post, making each trial 6 seconds long
    trial_spikes[trial_idx] = np.append(spikes_pre, spikes_post)

# Plot the spike times across trials
plot_rasters(trial_spikes, vline=3, title='Spikes per trial')

###################################################################################################
#
# Now that we have our simulated data, first we will calculate the firing rates across
# the pre- and post-event time ranges.
#
# To do so, we will use the :func:`~.compute_pre_post_rates` function.
#

###################################################################################################

# Define settings for the pre- and post- event time ranges
pre_window = [0, time_pre]
post_window = [time_pre, time_pre + time_post]

# Compute firing rates (Hz) pre- and post-event for all trials
frs_pre, frs_post = compute_pre_post_rates(trial_spikes, pre_window, post_window)

###################################################################################################
#
# Next, we can compute the t-test between the pre- and post-event firing rates across trials.
#
# To do this, we will use the :func:`~.compute_pre_post_ttest` function.
#

###################################################################################################

# Compute a t-test between the pre- and post-event firing rates
tval_t, pval_t = compute_pre_post_ttest(frs_pre, frs_post)

print('The t-value is {:1.3f}, with a p-value of {:.3e}'.format(tval_t, pval_t))

###################################################################################################
#
# In the above, we separately computed and then compared the firing rates around
# an event of interest.
#
# To compute average firing rates and compare them all together, we can use the
# :func:`~.compare_pre_post_activity` function.
#

###################################################################################################

# Calculate p-value and t-value between the firing rates using a t-test
# Also calculate the average firing rates pre- and post-event
avg_pre, avg_post, tval_t, pval_t = compare_pre_post_activity(trial_spikes, pre_window,
                                                              post_window, avg_type='mean')

# Print the average firing rates & t-test output
print('Average FR pre-event: {:1.3f}'.format(avg_pre))
print('Average FR post-event: {:1.3f}'.format(avg_post))

print('The t-value is {:1.3f}, with a p-value of {:.3e}'.format(tval_t, pval_t))

###################################################################################################
# Compare 1st and 2nd half of trials
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In real data, time or trial index may influence the outcome.
#
# For another example analysis, next we will examine the binned pre / post firing
# rates, split up by a split-half of the trials, to examine whether there is
# evidence for a difference in spiking activity across different trials.
#
# To do this, we can use the :func:`~.compare_trial_frs` function.
#

###################################################################################################

# Combine trials together into a 2d array
all_trials_data = np.array([frs_pre, frs_post]).T

# For each bin (pre and post event bins), compare 1st half of trials to 2nd half of trials
pre_between_halfs, post_between_halfs = compare_trial_frs(all_trials_data[:5], all_trials_data[5:])

# Print out the results
print('For pre bin comparison, the t-value is {:1.3f}, a p-value of {:.3e}.'.format(\
      pre_between_halfs[0], pre_between_halfs[1]))
print('For post bin comparison, the t-value is {:1.3f}, with a p-value of {:.3e}.'.format(\
      post_between_halfs[0], post_between_halfs[1]))

###################################################################################################
# Run 2-way ANOVA on multiple trials data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Another way of testing whether there are event-related changes in firing rates
# rate is to run a 2-way ANOVA.
#
# To do that, we will first organize the firing rates and the factors in a pandas DataFrame,
# which we will prepare for by organizing the data we need into a dictionary.
#

###################################################################################################

# Put the firing rates int a single array
pre_post_frs = np.concatenate([frs_pre, frs_post])

# Create a boolean array to differentiate between pre- and -post event
is_post_event = np.array([0] * len(frs_pre) + [1] * len(frs_pre))

# Create array with trial number information
trial_idx = np.concatenate([np.arange(0, len(frs_pre)), np.arange(0, len(frs_pre))])

# Put it all together into a dictionary
data = {'fr': pre_post_frs, 'is_post_event': is_post_event, 'trial_idx': trial_idx}

###################################################################################################
#
# We can now turn our data into a dataframe using the :func:`~.create_dataframe` function.
#

###################################################################################################

# Create the dataframe
df_pre_post = create_dataframe(data)
# Visualize the first entries of the dataframe
df_pre_post.head()

###################################################################################################
#
# Now that we have our data organized into a dataframe, we can run an ANOVA.
#
# Specifically, we will run an ANOVA to analyze whether there is an effect of the event on
# firing rates, as well as checking whether there is an effect of trial index on firing rate.
#
# To run the ANOVA, we need to define the formula we want to run.
#
# Then, using the dataframe and the formula, we can use
# the :func:`~.fit_anova` function to fit our ANOVA.
#
# Note that in this example, we did not evaluate the effect of the interaction.
#

###################################################################################################

# Define the formula for the ANOVA to fit
formula = 'fr ~ C(is_post_event)+C(trial_idx)'

# Run 2-way ANOVA without interaction on our dataframe
anova_pre_post = fit_anova(df_pre_post, formula, return_type='results', anova_type=2)

# Print out results
print('F-value, pre vs post: {:1.3f}'.format(anova_pre_post['F']['C(is_post_event)']))
print('P-value, pre vs post: {:.3e}'.format(anova_pre_post['PR(>F)']['C(is_post_event)']))
print('F-value, trial index: {:1.3f}'.format(anova_pre_post['F']['C(trial_idx)']))
print('P-value, trial index: {:.3e}'.format(anova_pre_post['PR(>F)']['C(trial_idx)']))

###################################################################################################
# Compute the empirical p-value and z-score from distribution of surrogates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we will use surrogate data to test whether our analysis reflects a significant effect.
#
# First we calculate the change in firing rate for an example trial (first trial).
#
# Here firing rate different is computed between pre- and post-even rates
# (delta = post firing rate - pre firing rate).
#

###################################################################################################

# Get firing rate difference between post and pre
# This will be the value we compute the empirical p-value and the z-score for
fr_diff = frs_post[0] - frs_pre[0]

###################################################################################################
#
# To compare the delta firing rate, we need to generate our distribution of surrogates.
#
# Thus, we will shuffle the data from the first trial (as an example) 100 times (using isis).
# For each of the shuffles, we will calculate the change in firing rate.
#

###################################################################################################

# Settings for shuffle the data
n_shuff = 100

# Shuffle the spike data, to be used to calculate surrogates
shuff_spikes = shuffle_isis(trial_spikes[0], n_shuffles=n_shuff)

###################################################################################################
#
# Now that we have shuffled spikes, we can compute our measure of interest on the
# surrogate data. This will give us a surrogate distribution which we can then use
# to compute the empirical p-value and the z-score, to examine if the effect in the
# real data is greater than expected by change (in the shuffles).
#

###################################################################################################

# Calculate surrogate measures
shuff_frs_pre, shuff_frs_post = compute_pre_post_rates(shuff_spikes, pre_window, post_window)
shuff_fr_diff = shuff_frs_post - shuff_frs_pre

print('Minimum delta FR across surrogates: {:1.3f}'.format(np.min(shuff_fr_diff)))
print('Maximum delta FR across surrogates: {:1.3f}'.format(np.max(shuff_fr_diff)))

###################################################################################################
#
# Now we can compute the empirical p-value and z-score of change in firing rate from
# the distribution of surrogates, using the :func:`~.compute_surrogate_stats` function.
#

###################################################################################################

# Calculate empirical p-value and z-score of difference in firing rate with respect to surrogates
surr_pval, surr_zscore = compute_surrogate_stats(fr_diff, shuff_fr_diff)

print('For the surrogate comparison, the z-score is {:1.3f}, and the p-value is {:.3e}'.format(\
      surr_zscore, surr_pval))

###################################################################################################
#
# Lastly, plot the distribution of surrogates with calculated delta firing rate.
#

###################################################################################################

# Plot distribution of surrogates, with calculated delta firing rate & p-value
plot_surrogates(shuff_fr_diff, fr_diff, surr_pval)

###################################################################################################
# Compute f-value from spiking data using 1-way ANOVA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this example, we will compute a statistical test of whether there is an effect
# of spatial position on neuron firing.
#
# To do so, we will simulate spike trains across trials (8 seconds) across spatial bins.
#
# The simulated data will then be organized (per trial, per bin) into dataframes,
# which we can then use to fit the ANOVA.
#
# For this example, we will generate a new set of spiking data.
# Note that this simulation matches the one used in the Spatial Analysis tutorial.
#

###################################################################################################

# Generate some positional data
x_pos = np.linspace(0, 15, 8000)
y_pos = np.linspace(0, 5, 8000)
position = np.array([x_pos, y_pos])

# Set number of spatial bins, 3 x-bins and 5 y-bins
bins = [3, 5]
n_bins = bins[0] * bins[1]

# Compute spatial bin edges
x_edges, y_edges = compute_bin_edges(position, bins)

# Set number of trials
n_trials = 10
bin_firing_all = np.zeros([n_trials, n_bins])

for ind in range(n_trials):

    # Simulate a spike train with a sampling rate of 1000 Hz
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
#
# Note that in this example we have a different data organization, due to the binning, and so
# in this case we will create a dataframe using the :func:`~.create_dataframe_bins` function.
#

###################################################################################################

# Organize spiking data into dataframe
df = create_dataframe_bins(bin_firing_all, dropna=True)
# Visualize the first entries of the dataframe
df.head()

###################################################################################################
#
# Once we have the dataframe, we can fit the ANOVA using the
# :func:`~.fit_anova` function, as above.
#

###################################################################################################

# Compute f_value from spiking data using ANOVA
f_val = fit_anova(df, 'fr ~ C(bin)', feature='C(bin)', return_type='f_val', anova_type=1)
print('F-value: {:.3}'.format(f_val))

###################################################################################################
#
# Note that the procedure we employed here could also be combined with what we did above
# to shuffle the data, in order to calculate the ANOVA F-value across surrogates and
# then compare to the real data.
#
