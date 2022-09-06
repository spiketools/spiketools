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
                                          compute_bin_counts_assgn, compute_bin_counts_pos,
                                          normalize_bin_counts, compute_occupancy)
from spiketools.spatial.utils import compute_pos_ranges, compute_bin_width, compute_bin_time
from spiketools.spatial.information import compute_spatial_information

# Import spiketrain simulation function
from spiketools.sim.train import sim_spiketrain_binom

# Import plotting functions
from spiketools.plts.spatial import plot_positions, plot_heatmap, plot_position_by_time
from spiketools.plts.utils import make_axes

###################################################################################################
#
# To start, we will first define some simulated data to work with
#

###################################################################################################

# Define some position data
x_pos = np.array([0, 1, 2, 3, 4, 3.2, 2.1, 2, 1, 0, 0, 0.1, 1, 1.5, 1, 1])
y_pos = np.array([0, 0, 0.1, 1, 1.5, 1, 1, 2.1, 2, 1, 0, 1, 2, 3, 4, 3.2])
position = np.array([x_pos, y_pos])

# Set the time width of each position sample
bin_widths = np.array([1, 1, 1, 2, 1.5, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0.5])

# Set number of spatial bins, 3 x-bins and 5 y-bins
bins = [3, 5]

# Define timestamps
timestamps = np.array([0, 1, 2, 3, 5, 6.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17])

###################################################################################################
# Compute and plot distances and speed using position
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Use x- and y-position to compute distance between two or more points,
# cumulative distance traveled and speed.
#

###################################################################################################

# Look at the range of our x and y positions
ranges = compute_pos_ranges(position)

print('The x-position ranges from {:1.1f} to {:1.1f}'.format(ranges[0][0], ranges[0][1]))
print('The y-position ranges from {:1.1f} to {:1.1f}'.format(ranges[1][0], ranges[1][1]))

###################################################################################################

# Plot positions (coordinates marked x are the actual points)
plot_positions(position, alpha=1, ls='-', marker='x', color='tab:gray', markersize=10,
               title='Tracking', xlabel='x-position', ylabel='y-position')
_ = plt.legend(['coordinates'])

###################################################################################################

# Plot x-position by time
plot_position_by_time(timestamps, x_pos, alpha=1, ls='-', marker='x', color='tab:gray',
                      markersize=10, title='X-position by time', xlabel='time', ylabel='x-position')
_ = plt.legend(['coordinates'])

# Plot y-position by time
plot_position_by_time(timestamps, y_pos, alpha=1, ls='-', marker='x', color='tab:gray',
                      markersize=10, title='Y-position by time', xlabel='time', ylabel='y-position')
_ = plt.legend(['coordinates'])

###################################################################################################
#
# With our position data, we can use some function to compute distance measures, including:
#
# - :func:`~.compute_distance`: which computes the distance between two points
# - :func:`~.compute_distances`: which computes distances across a sequence of positions
# - :func:`~.compute_cumulative_distances`: which computes cumulative distance across positions
#

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

# Plot distance and speed measures
ax1, ax2, ax3 = make_axes(3, 1, sharex=True, hspace=0.4, figsize=(8, 6))

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

###################################################################################################
# Divide position in spatial bin edges and plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we can start binning our position data.
#
# To do so, we will compute x- and y- spatial bin edges, using the
# :func:`~.compute_bin_edges`: function.
#
# Once we have our bin edges, we can compute the bin widths using the
# :func:`~.compute_bin_width`: function.
#

###################################################################################################

# Compute spatial bin edges
x_edges, y_edges = compute_bin_edges(position, bins)

# Compute the width of each spatial bin
x_bins_spatial_width = compute_bin_width(x_edges)
y_bins_spatial_width = compute_bin_width(y_edges)
print('The x spatial bins have width = {:.3}'.format(x_bins_spatial_width))
print('The y spatial bins have width = {:.3}'.format(y_bins_spatial_width))

###################################################################################################
#
# Now that we have created our bin definition, we can plot the spatial grid.
#

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
# Next, we will compute the spatial bin assignment for the position data using the
# previously computed spatial bin edges.
#
# To do so, we will use the :func:`~.compute_bin_assignment`: function.
#

###################################################################################################

# Now let us check where the first 7 position data points using the same x_edges and y_edges
n_points = 7
x_bins, y_bins = compute_bin_assignment(position[:, :n_points], x_edges, y_edges)

# We can check they match the positions in plot (ii)
for ind in range(0, n_points):
    print('The point (x, y) = ({:1.1f}, {:1.1f}) is in x_bin {:1d} and y_bin {:1d}.'.format(\
          position[0, ind], position[1, ind], x_bins[ind], y_bins[ind]))

###################################################################################################
# Compute time in each timestamp sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute time width of the timestamp sampling bins, using the
# :func:`~.compute_bin_time`: function.
#

###################################################################################################

# Let us now compute the time in each timestamp bin
bin_time = compute_bin_time(timestamps)
print('The time widths of the sampling bins are: ', bin_time)

###################################################################################################
# Compute and plot occupancy and position counts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we are interested in how much time was spent in each bin of the spatial grid.
#
# To measure this, we will compute the occupancy across the spatial bins, using
# the :func:`~.compute_occupancy`: function.
#
# For 2D case, compute occupancy using position, timestamps, bins, speed, and a speed threshold.
# Also compute 2D position bin occurrence counts (no speed thresholding).
# Plot heatmaps of 2D occupancy and 2D position bin occurrence counts.
#

###################################################################################################

# Update speed to match length of position data
speeds = np.insert(speeds, 0, 0)

# Define speed threshold, used to position values for speed less than the threshold
speed_thresh =.5e-3

# Compute the 2D occupancy
occupancy = compute_occupancy(position, timestamps, bins,
                              speed=speeds, speed_threshold=speed_thresh)

# Plot the compute 2D occupancy
plot_heatmap(occupancy, cbar=True,
             title='Occupancy heatmap w/ speed threshold')

###################################################################################################
#
# Another way to explore occupancy measures is to check the number of
# occurrences with each spatial bin.
#
# This can be computed with the :func:`~.compute_bin_counts_pos`: function.
#

###################################################################################################

# Compute the 2D position bin occurrence counts
bin_counts_pos = compute_bin_counts_pos(position, bins)

# Plot the 2D position bin occurrence counts
plot_heatmap(bin_counts_pos, cbar=True,
             title='Position bin occurrence counts heatmap')

###################################################################################################
#
# We can also compute occupancy for 1D data.
#
# For this examples, we will compute 1D occupancy using the x-position data with the
# same timestamps, x-bins, speed, and speed threshold as before.
#

###################################################################################################

# Compute the 1D occupancy
occupancy_1d = compute_occupancy(position[0], timestamps, bins[0],
                                 speed=speeds, speed_threshold=speed_thresh)

# Plot the 1D occupancy
plot_heatmap(occupancy_1d, title='X-occupancy heatmap w/ speed threshold')

###################################################################################################
#
# As before, we can also check the bin occurrence counts for the 1D data.
#

###################################################################################################

# Compute and plot 1D position bin occurrence counts for x-position
bin_counts_pos_1d = compute_bin_counts_pos(position[0], bins[0])
plot_heatmap(bin_counts_pos_1d, title='X-position bin occurrence counts heatmap')

###################################################################################################
# Compute 2D and 1D spatial information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we want to measure to relationship between spiking and position data.
#
# To do so, we will compute a spatial information measure between simulated spiking data
# and associated position data.
#
# To do so, we will use the :func:`~.compute_spatial_information`: function.
#

###################################################################################################

# Simulate a spike train with chance level
spike_train = sim_spiketrain_binom(0.5, n_samples=len(x_pos))

# Get spike position bins, and the x and y position bins corresponding to spike positions
spike_bins = np.where(spike_train == 1)[0]
spike_x, spike_y = compute_bin_assignment(position[:, spike_bins], x_edges, y_edges,
                                          include_edge=True)

###################################################################################################
#
# Let's first calculate the 1D spatial information, using only the x-dimension.
#
# To do so, we will compute the bin firing, using the
# :func:`~.compute_bin_counts_assgn`: function to compute the number of spikes per bin.
#
# We then need to normalize this measure by occupancy, which we can do with the
# :func:`~.normalize_bin_counts`: function.
#

###################################################################################################

# Compute bin firing and normalize it
bin_firing_1d = compute_bin_counts_assgn(bins=[bins[0]], xbins=spike_x, occupancy=occupancy_1d)
normalized_bin_fr_1d = normalize_bin_counts(bin_firing_1d, occupancy=occupancy_1d)

###################################################################################################
#
# We are now ready to compute the spatial information.
#

###################################################################################################

# Compute 1d spatial information
spatial_information_1d = compute_spatial_information(normalized_bin_fr_1d, occupancy_1d,
                                                     normalize=False)
print('The 1D spatial information is = {:.3}'.format(spatial_information_1d))

###################################################################################################
#
# We can also follow the same procedure to compute spatial information for 2D data.
#

###################################################################################################

# Compute bin firing and normalize it
bin_firing = compute_bin_counts_assgn(bins=bins, xbins=spike_x, ybins=spike_y)
normalized_bin_fr = normalize_bin_counts(bin_firing, occupancy=occupancy)

# Compute 2d spatial information
spatial_information_2d = compute_spatial_information(normalized_bin_fr, occupancy, normalize=False)
print('The 2D spatial information is = {:.3}'.format(spatial_information_2d))

###################################################################################################
