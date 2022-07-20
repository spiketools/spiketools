"""
Spatial Analyses
================

Analyze spatial data and apply spatial measures to spiking data.

This tutorial primarily covers the ``spiketools.spatial`` module.
"""

###################################################################################################
# Apply spatial measures to spatial data
# --------------------------------------
#
# Sections
# ~~~~~~~~
#
# This tutorial contains the following sections:
#
# 1. Compute and plot distances and speed using position
# 2. Divide position in spatial bin edges
# 3. Compute spatial bin assignment using spatial bin edges
# 4. Compute time in each timestamp sample
# 5. Compute and plot occupancy and position counts
# 6. Compute 2D and 1D spatial information
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 1

# Import auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

# Import functions from spiketools.spatial
from spiketools.spatial.position import (compute_distance, compute_distances,
                                         compute_cumulative_distances, compute_speed)
from spiketools.spatial.occupancy import (compute_bin_edges, compute_bin_assignment,
                                          compute_bin_time, compute_occupancy,
                                          compute_bin_counts_assgn, normalize_bin_counts,
                                          compute_bin_counts_pos)
from spiketools.spatial.utils import compute_pos_ranges, compute_bin_width
from spiketools.spatial.information import compute_spatial_information

# Import spiketrain simulation function
from spiketools.sim.train import sim_spiketrain_binom

# Import plotting functions
from spiketools.plts.spatial import (plot_positions, plot_heatmap, plot_trial_heatmaps,
                                     plot_position_by_time)

###################################################################################################

###################################################################################################

# Set some tracking data
x_pos = np.array([0, 1, 2, 3, 4, 3.2, 2.1, 2, 1, 0, 0, 0.1, 1, 1.5, 1, 1])
y_pos = np.array([0, 0, 0.1, 1, 1.5, 1, 1, 2.1, 2, 1, 0, 1, 2, 3, 4, 3.2])
position = np.array([x_pos, y_pos])

# Set the time width of each position sample
bin_widths = np.array([1, 1, 1, 2, 1.5, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0.5])

# Set number of spatial bins, 3 x-bins and 5 y-bins
bins = [3, 5]

# Set timestamp
timestamps = np.array([0, 1, 2, 3, 5, 6.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17])

###################################################################################################
# Compute and plot distances and speed using position
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Use x- and y-position to compute distance between two or more points, cumulative distance
# traveled and speed
#

###################################################################################################

# Look at the range of our x and y positions
ranges = compute_pos_ranges(position)
print(f'The x-position ranges from {ranges[0][0]} to {ranges[0][1]}')
print(f'The y-position ranges from {ranges[1][0]} to {ranges[1][1]}')

###################################################################################################

# Plot positions (coordinates marked x are the actual points)
plot_positions(position, alpha=1, ls='-', marker='x', color='tab:gray', markersize=10,
               title='Tracking', xlabel='x-position', ylabel='y-position')
_ = plt.legend(['coordinates'])

###################################################################################################

# Plot x-position by time
plot_position_by_time(timestamps, x_pos, alpha=1, ls='-', marker='x', color='tab:gray', markersize=10,
               title='X-position by time', xlabel='time', ylabel='x-position')
_ = plt.legend(['coordinates'])

# Plot y-position by time
plot_position_by_time(timestamps, y_pos, alpha=1, ls='-', marker='x', color='tab:gray', markersize=10,
               title='Y-position by time', xlabel='time', ylabel='y-position')
_ = plt.legend(['coordinates'])

###################################################################################################

# Compute distance between start and end point
dist_start_end = compute_distance(x_pos[0], y_pos[0], x_pos[-1], y_pos[-1])

# Compute distance traveled at each point
dist_traveled = compute_distances(x_pos, y_pos)

# Compute total distance traveled
cumulative_dist_traveled = compute_cumulative_distances(x_pos, y_pos)

# Compute speed at each point
bin_widths = np.array([1, 1, 1, 2, 1.5, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0.5])
speeds = compute_speed(x_pos, y_pos, bin_widths)

###################################################################################################

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# Plot distance traveled at each time
plot_position_by_time(timestamps[1:], dist_traveled,
               ax=ax1, alpha=1, ls='-', marker='x', color='tab:pink', markersize=10,
               title='Distance traveled at each point',
               xlabel='time (t)', ylabel='speed (u/t)')

# Plot cumulative distance traveled per time
plot_position_by_time(timestamps[1:], cumulative_dist_traveled,
               ax=ax2, alpha=1, ls='-', marker='x', color='tab:olive', markersize=10,
               title='Cumulative distance traveled at each point',
               xlabel='time (t)', ylabel='speed (u/t)')

# Plot speed at each time point
plot_position_by_time(timestamps[1:], speeds,
               ax=ax3, alpha=1, ls='-', marker='x', color='tab:cyan', markersize=10,
               title='Speed at each point', xlabel='time (t)', ylabel='speed (u/t)')

# Add padding between subplots, and make figure bigger
fig.tight_layout(pad=0.05)
fig.set_size_inches((15/2.54, 20/2.54))

###################################################################################################
# Divide position in spatial bin edges and plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute x- and y- spatial bin edges and plot spatial grid.
#

###################################################################################################

# Compute spatial bin edges
x_edges, y_edges = compute_bin_edges(position, bins)

# Compute the width of each spatial bin
x_bins_spatial_width = compute_bin_width(x_edges)
y_bins_spatial_width = compute_bin_width(y_edges)
print(f'The x spatial bins have width = {x_bins_spatial_width}')
print(f'The y spatial bins have width = {y_bins_spatial_width}')

###################################################################################################

# Plot grid of spatial bins with tracking on top
plot_positions(position, x_bins=x_edges, y_bins=y_edges,
               alpha=1, ls='-', marker='x', color='tab:gray', markersize=10,
               title='Tracking and spatial bins', xlabel='x-position', ylabel='y-position')
_ = plt.legend(['Tracking'], loc='upper left')

###################################################################################################
# Compute spatial bin assignment using spatial bin edges
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute spatial bin assignment for position using previously computed spatial bin edges.
#

###################################################################################################

# Now let us check where the 1st 7 point of position using the same x_edges and y_edges
n_points = 7
x_bins, y_bins = compute_bin_assignment(position[:, :n_points], x_edges, y_edges)

# We can check they match the positions in plot (ii)
for ind in range(0, n_points):
    print(f'The point (x, y) = ({position[0, ind]}, {position[1, ind]}) is in the x_bin \
          {x_bins[ind]}, and on the y_bin {y_bins[ind]}.')

###################################################################################################
# Compute time in each timestamp sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute time width of the timestamp sampling bins.
#

###################################################################################################

# Let us now compute the time in each timestamp bin
bin_time = compute_bin_time(timestamps)
print(f'The time widths of the the sampling bins are: {bin_widths}')

###################################################################################################
# Compute and plot occupancy and position counts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we are interested in how much time was spent in each bin of the spatial grid (each
# sub-region of the space).
# For 2D case, compute occupancy using position, timestamps, bins, speed, and a speed threshold.
# Also compute 2D position bin occurrence counts (no speed thresholding).
# Plot heatmaps of 2D occupancy and 2D position bin occurrence counts.
#
# For 1D case, compute occupancy using x-position, timestamps, x-bins, speed, and a speed 
# threshold.
# Also compute 1D x-position bin occurrence counts (no speed thresholding).
# Plot heatmaps of 1D occupancy and 1D x-position bin occurrence counts.
#

###################################################################################################

# Compute and plot 2D occupancy using the previously defined position, timestamps, bins and speed
# Set speed threshold, so that we ignore all occurrences with speed less than the threshold
speed_thresh=.5e-3
occupancy = compute_occupancy(position, timestamps, bins,
                              speed=np.insert(speeds, 0, 0), speed_threshold=speed_thresh)
plot_heatmap(occupancy, title='Occupancy heatmap w/ speed threshold')

# Compute and plot 2D position bin occurrence counts
bin_counts_pos = compute_bin_counts_pos(position, bins)
plot_heatmap(bin_counts_pos, title='Position bin occurrence counts heatmap')

# Compute and plot 1D occupancy using the previously defined x-position, timestamps, x-bins and
# speed
occupancy_1d = compute_occupancy(position[0], timestamps, bins[0],
                                 speed=np.insert(speeds, 0, 0), speed_threshold=speed_thresh)
plot_heatmap(occupancy_1d, title='X-occupancy heatmap w/ speed threshold')

# Compute and plot 1D position bin occurrence counts for x-position
bin_counts_pos_1d = compute_bin_counts_pos(position[0], bins[0])
plot_heatmap(bin_counts_pos_1d, title='X-position bin occurrence counts heatmap')

###################################################################################################
# Compute 2D and 1D spatial information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute spatial information for simulated spike positions, spatial bins, and occupancy.
#

###################################################################################################

# Simulate a spike train with chance level
spike_train = sim_spiketrain_binom(0.5, n_samples=len(x_pos))

# Get spike position bins
spike_bins = np.where(spike_train == 1)[0]
# Get x and y position bins corresponding to spike positions
spike_x, spike_y = compute_bin_assignment(position[:, spike_bins], x_edges, y_edges,
                                          include_edge=True)

###################################################################################################

# Calculate the 1D spatial information (x-dimension only)
# Compute bin firing and normalize it
bin_firing_1d = compute_bin_counts_assgn(bins=[bins[0]], xbins=spike_x, occupancy=occupancy_1d)
normalized_bin_fr_1d = normalize_bin_counts(bin_firing_1d, occupancy=occupancy_1d)
# Compute 1d spatial information
spatial_information_1d = compute_spatial_information(normalized_bin_fr_1d, occupancy_1d,
                                                     normalize=False)
print(f'The 1D spatial information is = {spatial_information_1d}')

###################################################################################################

# Compute the 2D spatial information for spikes
# Compute bin firing and normalize it
bin_firing = compute_bin_counts_assgn(bins=bins, xbins=spike_x, ybins=spike_y)
normalized_bin_fr = normalize_bin_counts(bin_firing, occupancy=occupancy)
# Compute 2d spatial information
spatial_information_2d = compute_spatial_information(normalized_bin_fr, occupancy, normalize=False)
print(f'The 2D spatial information is = {spatial_information_2d}')

###################################################################################################
