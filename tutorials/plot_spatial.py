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
# 5. Compute and plot occupancy
# 6. Compute 2D and 1D spatial information
#

###################################################################################################

# sphinx_gallery_thumbnail_number = 1

# import auxiliary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import all functions from spiketools.spatial
from spiketools.spatial.position import *
from spiketools.spatial.occupancy import *
from spiketools.spatial.utils import *
from spiketools.spatial.information import *

# import sim_spiketrain_binom to simulate spiketrain
from spiketools.sim.dist import sim_spiketrain_binom

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
# ~~~~~~~~
#
# Use x- and y-position to compute distance between two or more points, cumulative distance 
# traveled and speed
#

###################################################################################################

# Look at the range of our x and y positions
ranges = get_pos_ranges(position)
print(f'The x-position ranges from {ranges[0][0]} to {ranges[0][1]}')
print(f'The y-position ranges from {ranges[1][0]} to {ranges[1][1]}')

# Plot positions (coordinates marked x are the actual points)
plt.plot(x_pos, y_pos, '-x', c = 'tab:gray', markersize = 10)
plt.title('Tracking')
plt.xlabel('x-position')
plt.ylabel('y-position')
plt.legend(['coordinates'])

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

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# Plot distance traveled at each time
ax1.plot(dist_traveled, '-x', c = 'tab:pink', markersize = 10)
ax1.set_title('Distance traveled at each point')
ax1.set_xlabel('time (t)')
ax1.set_ylabel('distance (u)')

# Plot cumulative distance traveled per time
ax2.plot(cumulative_dist_traveled, '-x', c = 'tab:olive', markersize = 10)
ax2.set_title('Cumulative distance traveled at each point')
ax2.set_xlabel('time (t)')
ax2.set_ylabel('distance (u)')

# Plot speed at each time point
ax3.plot(speeds, '-x', c = 'tab:cyan', markersize = 10)
ax3.set_title('Speed at each point')
ax3.set_xlabel('time (t)')
ax3.set_ylabel('speed (u/t)')

# Add some padding between subplots
f.tight_layout(pad = 0.05)
# Make figure bigger
f.set_size_inches((15/2.54, 20/2.54))

###################################################################################################
# 2. Divide position in spatial bin edges and plot
# ~~~~~~~~
#
# Compute x- and y- spatial bin edges and plot spatial grid. 
#

###################################################################################################

x_edges, y_edges = compute_spatial_bin_edges(position, bins)

# Compute the width of each spatial bin
x_bins_spatial_width = get_bin_width(x_edges)
y_bins_spatial_width = get_bin_width(y_edges)
print(f'The x spatial bins have width = {x_bins_spatial_width}')
print(f'The y spatial bins have width = {y_bins_spatial_width}')

###################################################################################################

# Plot: (i) grid of spatial bins, and (ii) grid of spatial bins with tracking on top
f, (ax1, ax2) = plt.subplots(1, 2)
# Plot spatial bins in ax1 and ax2
for i in range(0, max(len(x_edges), len(y_edges))):
    if i < len(x_edges):
        l1, = ax1.plot([x_edges[i], x_edges[i]], [y_edges[0], y_edges[-1]], c = 'tab:orange')
        ax2.plot([x_edges[i], x_edges[i]], [y_edges[0], y_edges[-1]], c = 'tab:orange')
    if i < len(y_edges):
        ax1.plot([x_edges[0], x_edges[-1]], [y_edges[i], y_edges[i]], c = 'tab:orange')
        ax2.plot([x_edges[0], x_edges[-1]], [y_edges[i], y_edges[i]], c = 'tab:orange')
        
# Set title and axis labels for left plot (plot (i))
ax1.set_title('(i) Visualization of spatial bins')
ax1.set_xlabel('x-position')
ax1.set_ylabel('y-position')        

# Add position plot to right plot (plot (ii)), for visualization of how the tracking fits in the bins
track = ax2.plot(x_pos, y_pos, '-x', c = 'tab:gray', markersize = 10)
ax2.set_title('(ii) Tracking and spatial bins')
ax2.legend(['spatial bins','Tracking'], loc = 'upper left')
hl_dict = {handle.get_label(): handle for handle in ax2.get_legend().legendHandles}
hl_dict['_line1'].set_color('tab:gray')
hl_dict['_line1'].set_marker('x')
ax2.set_xlabel('x-position')
ax2.set_ylabel('y-position')

# Add some padding between subplots
f.tight_layout(pad = 0.5)
# Make figure bigger
f.set_size_inches((20/2.54, 10/2.54))

###################################################################################################
# 3. Compute spatial bin assignment using spatial bin edges
# ~~~~~~~~
#
# Compute spatial bin assignment for position using previously computed spatial bin edges.
#

###################################################################################################

# Now let us check where the 1st 7 point of position using the same x_edges and y_edges
n_points = 7
x_bins, y_bins = compute_spatial_bin_assignment(position[:, :n_points], x_edges, y_edges, include_edge=True)
# We can check they match the positions in plot (ii)
for i in range (0, n_points):
    print(f'The point (x, y) = ({position[0, i]}, {position[1, i]}) is in the x_bin {x_bins[i]}, and on the y_bin {y_bins[i]}.')

###################################################################################################
# 4. Compute time in each timestamp sample
# ~~~~~~~~
#
# Compute time width of the timestamp sampling bins.
#

###################################################################################################

# Let us now compute the time in each timestamp bin
bin_time = compute_bin_time(timestamps)
print(f'The time widths of the the sampling bins are: {bin_widths}')

###################################################################################################
# 5. Compute and plot occupancy
# ~~~~~~~~
#
# Compute occupancy using position, timestamps, bins and speed. Plot a heatmap of occupancy.
#

###################################################################################################

# Compute occupancy using the previously defined position, timestamps, bins and speed
occupancy = compute_occupancy(position, timestamps, bins, speed=np.insert(speeds, 0, 0))

# Plot occupancy using a heatmap
sns.heatmap(occupancy)
plt.title('Occupancy heatmap')
plt.xlabel('x spatial bin')
plt.xlabel('y spatial bin')

###################################################################################################
# 6. Compute 2D and 1D spatial information
# ~~~~~~~~
#
# Compute spatial information for simulated spike positions, spatial bins, and occupancy.
#

###################################################################################################

# Calculate the spatial information for the x-dimension only
data = spike_x
x_occupancy = np.sum(occupancy, axis = 1)
spatial_information_1d = compute_spatial_information_1d(data, x_occupancy, bins)
print(f'The 1D spatial information is = {spatial_information_1d}')

###################################################################################################

# Compute the 2D spatial information for spikes
# Simulate a spike train with chance level
spike_train = sim_spiketrain_binom(0.5, n_samples=len(x_pos))
# Get x and y positions corresponding
spike_x = x_pos[np.where(spike_train == 1)]
spike_y = y_pos[np.where(spike_train == 1)]

spatial_information_2d = compute_spatial_information_2d(spike_x, spike_y, bins, occupancy)
print(f'The 2D spatial information is = {spatial_information_2d}')

###################################################################################################
