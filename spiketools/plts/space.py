"""Plots for spatial measures and analyses."""

from copy import deepcopy

import numpy as np

from spiketools.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
def plot_space_heat(data, transpose=False, ignore_zero=False, title=None, figsize=None, ax=None):
    """Plot a spatial heat map.

    Parameters
    ----------
    data : 2d array
        Measure to plot across a grided environment.
    transpose : bool, optional, default: False
        Whether to transpose the data before plotting.
    ignore_zero : bool, optional, default: False
        If True, replaces 0's with NaN for plotting.
    title : str, optional
        Title to add to the figure.
    figsize : list, optional
        Size to create the figure.
    ax : Axes, optional
        Axis object upon which to plot.
    """

    ax = check_ax(ax, figsize=figsize)

    if transpose:
        data = data.T

    if ignore_zero:
        data = deepcopy(data)
        data[data == 0.] = np.nan

    ax.imshow(data)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)
