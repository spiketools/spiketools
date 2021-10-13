"""Plots for spatial measures and analyses."""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from spiketools.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
def plot_space_heat(data, transpose=False, smooth=False, smoothing_kernel=1.5,
                    ignore_zero=False, cbar=False, cmap=None, vmin=None, vmax=None,
                    title=None, ax=None, **plt_kwargs):
    """Plot a spatial heat map.

    Parameters
    ----------
    data : 2d array
        Measure to plot across a grided environment.
    transpose : bool, optional, default: False
        Whether to transpose the data before plotting.
    smooth : bool, optional, default: False
        Whether to smooth the data before plotting.
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

    if title:
        ax.set_title(title)

    if cbar:
        colorbar = plt.colorbar(im)
        colorbar.outline.set_visible(False)


def _smooth_data(data, sigma):
    """Smooth data for plotting."""

    data = deepcopy(data)
    data[np.isnan(data)] = 0

    data = gaussian_filter(data, sigma=sigma)

    return data
