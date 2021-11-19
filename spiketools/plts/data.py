"""Plots for various data."""

from itertools import repeat

import numpy as np

from spiketools.utils.select import get_avg_func

from spiketools.plts.utils import check_ax, savefig, set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_lines(x_values, y_values, vline=None, ax=None, **plt_kwargs):
    """Plot data as a line.

    Parameters
    ----------
    x_values : 1d or 2d array or list
        Data to plot on the x and y axis.
    ax : Axes, optional
        Axis object upon which to plot.
    line : float or list, optional, default: 0
        Position(s) to draw a vertical line. If None, no line is drawn.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    x_values = repeat(x_values) if (isinstance(x_values, np.ndarray) and x_values.ndim == 1) else x_values
    y_values = [y_values] if (isinstance(y_values, np.ndarray) and y_values.ndim == 1) else y_values

    for x_vals, y_vals in zip(x_values, y_values):
        ax.plot(x_vals, y_vals, **plt_kwargs)

    if vline is not None:
        vline = [vline] if isinstance(vline, (int, float)) else vline
        for line in vline:
            ax.axvline(line)


@savefig
@set_plt_kwargs
def plot_hist(data, average=None, ax=None, **plt_kwargs):
    """Plot data as a histogram.

    Parameters
    ----------
    data : 1d array
        Data to plot.
    average : {'median', 'mean'}, optional
        Which kind of average to compute and add to the plot.
        If None, no average is plotted.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.

    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    ax.hist(data, **plt_kwargs)

    if average:
        ax.axvline(get_avg_func(average)(data), lw=4, color='red', alpha=0.8)


@savefig
@set_plt_kwargs
def plot_bar(data, labels=None, ax=None, **plt_kwargs):
    """Plot data in a bar graph.

    Parameters
    ----------
    data : list of float
        Data to plot.
    labels : list of str
        Labels for the bar plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    if not labels:
        labels = ['d' + str(ind) for ind in range(len(data))]

    ax.bar(labels, data, **plt_kwargs)
    ax.set(xlim=[-0.5, len(data)-0.5])
