"""Plots for spatial measures and analyses."""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from spiketools.plts.utils import check_ax, savefig, set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_positions(positions, spike_positions=None, x_bins=None, y_bins=None,
                   ax=None, **plt_kwargs):
    """Plot positions.

    Parameters
    ----------
    positions : 2d array
        Position data.
    spike_positions : 2d array, optional
        Positions values at which spikes occur.
        If provided, these are added to the plot as red dots.
    x_bins, y_bins : list of float
        Bin edges for each axis.
        If provided, these are used to draw grid lines on the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    ax.plot(*positions, alpha=plt_kwargs.pop('alpha', 0.35), **plt_kwargs)

    if spike_positions is not None:
        ax.plot(spike_positions[0, :], spike_positions[1, :],
                '.', color='red', alpha=0.35, ms=6)

    if x_bins is not None:
        ax.set_xticks(x_bins, minor=False)
    if y_bins is not None:
        ax.set_yticks(y_bins, minor=False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    if x_bins is not None or y_bins is not None:
        ax.grid()


@savefig
@set_plt_kwargs
def plot_space_heat(data, transpose=False, smooth=False, smoothing_kernel=1.5,
                    ignore_zero=False, cbar=False, cmap=None, vmin=None, vmax=None,
                    ax=None, **plt_kwargs):
    """Plot a spatial heat map.

    Parameters
    ----------
    data : 2d array
        Measure to plot across a grided environment.
    transpose : bool, optional, default: False
        Whether to transpose the data before plotting.
    smooth : bool, optional, default: False
        Whether to smooth the data before plotting.
    smoothing_kernel : float, optional, default: 1.5
        Standard deviation of the gaussian kernel to apply for smoothing.
    ignore_zero : bool, optional, default: False
        If True, replaces 0's with NaN for plotting.
    cbar : bool, optional, default: False
        Whether to add a colorbar to the plot.
    cmap : str, optional
        Which colormap to use to plot.
    vmin, vmax : float, optional
        Min and max plot ranges.
    title : str, optional
        Title to add to the figure.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    if transpose:
        data = data.T

    if smooth:
        data = _smooth_data(data, smoothing_kernel)

    if ignore_zero:
        data = deepcopy(data)
        data[data == 0.] = np.nan

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **plt_kwargs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    if cbar:
        colorbar = plt.colorbar(im)
        colorbar.outline.set_visible(False)


def _smooth_data(data, sigma):
    """Smooth data for plotting, using a gaussian kernel.

    Parameters
    ----------
    data : 2d array
        Data to smooth.
    sigma : float
        Standard deviation of the gaussian kernel to apply for smoothing.

    Returns
    -------
    data : 2d array
        The smoothed data.
    """

    data = deepcopy(data)
    data[np.isnan(data)] = 0

    data = gaussian_filter(data, sigma=sigma)

    return data
