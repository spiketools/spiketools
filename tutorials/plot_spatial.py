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
# This tutorial contains the following sections:
#
# 1. Compute and plot distances and speed using position
# 2. Divide position in spatial bin edges
# 3. Compute spatial bin assignment using spatial bin edges
# 4. Compute time in each timestamp sample
# 5. Compute and plot occupancy
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
                                          compute_bin_firing, normalize_bin_firing)
from spiketools.spatial.utils import get_pos_ranges, get_bin_width
from spiketools.spatial.information import compute_spatial_information

# Import spiketrain simulation function
from spiketools.sim.train import sim_spiketrain_binom

# Import plotting functions
from spiketools.plts.spatial import plot_positions, plot_heatmap

###################################################################################################

###################################################################################################

# Set some tracking data
x_pos = np.linspace(0, 15, 16)
y_pos = np.array([0, 0, 0.1, 1, 1.5, 1, 1, 2.1, 2, 1, 0, 1, 2, 3, 4, 3.2])
position = np.array([x_pos, y_pos])

# Set the time width of each position sample
bin_widths = np.array([1, 1, 1, 2, 1.5, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0.5])

# Set number of spatial bins, 3 x-bins and 5 y-bins
bins = [3, 5]

# Set timestamp
timestamps = np.array([0, 1, 2, 3, 5, 6.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17])

###################################################################################################
# 1. Compute and plot distances and speed using position
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Use x- and y-position to compute distance between two or more points, cumulative distance
# traveled and speed
#

###################################################################################################

# Look at the range of our x and y positions
ranges = get_pos_ranges(position)
print(f'The x-position ranges from {ranges[0][0]} to {ranges[0][1]}')
print(f'The y-position ranges from {ranges[1][0]} to {ranges[1][1]}')

###################################################################################################

# Plot positions (coordinates marked x are the actual points)
plot_positions(position, alpha=1, ls='-', marker='x', color='tab:gray', markersize=10,
               title='Tracking', xlabel='x-position', ylabel='y-position')
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
# plot distance traveled at each time
plot_positions(np.append([x_pos[1:]], [dist_traveled], axis=0),
               ax=ax1, alpha=1, ls='-', marker='x', color='tab:pink', markersize=10,
               title='Distance traveled at each point',
               xlabel='time (t)', ylabel='speed (u/t)')

# plot cumulative distance traveled per time
plot_positions(np.append([x_pos[1:]], [cumulative_dist_traveled], axis=0),
               ax=ax2, alpha=1, ls='-', marker='x', color='tab:olive', markersize=10,
               title='Cumulative distance traveled at each point',
               xlabel='time (t)', ylabel='speed (u/t)')

# plot speed at each time point
plot_positions(np.append([x_pos[1:]], [speeds], axis=0),
               ax=ax3, alpha=1, ls='-', marker='x', color='tab:cyan', markersize=10,
               title='Speed at each point', xlabel='time (t)', ylabel='speed (u/t)')

# Add padding between subplots, and make figure bigger
fig.tight_layout(pad=0.05)
fig.set_size_inches((15/2.54, 20/2.54))

###################################################################################################
# 2. Divide position in spatial bin edges and plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute x- and y- spatial bin edges and plot spatial grid.
#

###################################################################################################

# Compute spatial bin edges
x_edges, y_edges = compute_bin_edges(position, bins)

# Compute the width of each spatial bin
x_bins_spatial_width = get_bin_width(x_edges)
y_bins_spatial_width = get_bin_width(y_edges)
print(f'The x spatial bins have width = {x_bins_spatial_width}')
print(f'The y spatial bins have width = {y_bins_spatial_width}')

###################################################################################################

# Plot grid of spatial bins with tracking on top
plot_positions(position, x_bins=x_edges, y_bins=y_edges,
               alpha=1, ls='-', marker='x', color='tab:gray', markersize=10,
               title='Tracking and spatial bins', xlabel='x-position', ylabel='y-position')
_ = plt.legend(['Tracking'], loc='upper left')

###################################################################################################
# 3. Compute spatial bin assignment using spatial bin edges
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# 4. Compute time in each timestamp sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute time width of the timestamp sampling bins.
#

###################################################################################################

# Let us now compute the time in each timestamp bin
bin_time = compute_bin_time(timestamps)
print(f'The time widths of the the sampling bins are: {bin_widths}')

###################################################################################################
# 5. Compute and plot occupancy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we are interested in how much time was spent in each bin of the spatial grid (each
# sub-region of the space).
# For 2D case, compute occupancy using position, timestamps, bins and speed.
# Also plot a heatmap of 2D occupancy.
# For 1D case, compute occupancy using x-position, timestamps, x-bins and speed.
# Also plot a heatmap of 1D occupancy.
#

###################################################################################################

# Compute and plot 2D occupancy using the previously defined position, timestamps, bins and speed
occupancy = compute_occupancy(position, timestamps, bins, speed=np.insert(speeds, 0, 0), 
                              speed_thresh=.5e-5)
plot_heatmap(occupancy, transpose=True, title='Occupancy heatmap')

# Compute and plot 1D occupancy using the previously defined x-position, timestamps, x-bins and 
# speed
occupancy_1d = compute_occupancy(position[0], timestamps, [bins[0]], speed=np.insert(speeds, 0, 0), 
                                 speed_thresh=.5e-5)
plot_heatmap(np.expand_dims(occupancy_1d, 1), transpose=True, title='X-Occupancy heatmap')

###################################################################################################
# 6. Compute 2D and 1D spatial information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
bin_firing_1d = compute_bin_firing(bins=[bins[0]], xbins=spike_x, occupancy=occupancy_1d)
normalized_bin_fr_1d = normalize_bin_firing(bin_firing_1d, occupancy=occupancy_1d)
# Compute 1d spatial information
spatial_information_1d = compute_spatial_information(normalized_bin_fr_1d, occupancy_1d, 
                                                     normalize=False)
print(f'The 1D spatial information is = {spatial_information_1d}')

###################################################################################################

# Compute the 2D spatial information for spikes
# Compute bin firing and normalize it
bin_firing = compute_bin_firing(bins=bins, xbins=spike_x, ybins=spike_y)
normalized_bin_fr = normalize_bin_firing(bin_firing, occupancy=occupancy)
# Compute 2d spatial information
spatial_information_2d = compute_spatial_information(normalized_bin_fr, occupancy, normalize=False)
print(f'The 2D spatial information is = {spatial_information_2d}')
