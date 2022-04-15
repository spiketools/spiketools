"""Plots for various data."""

from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt

from spiketools.measures.circular import bin_circular
from spiketools.utils.select import get_avg_func
from spiketools.plts.annotate import _add_vlines
from spiketools.plts.utils import check_ax, savefig, set_plt_kwargs
from spiketools.plts.settings import TEXT_SETTINGS

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_lines(x_values, y_values, vline=None, ax=None, **plt_kwargs):
    """Plot data as a line.

    Parameters
    ----------
    x_values, y_values : 1d or 2d array or list of 1d array
        Data to plot on the x and y axis.
    vline : float or list, optional
        Position(s) to draw a vertical line. If None, no line is drawn.
    ax : Axes, optional
        Axis object upon which to plot.
    line : float or list, optional, default: 0
        Position(s) to draw a vertical line. If None, no line is drawn.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    x_values = repeat(x_values) if (isinstance(x_values, np.ndarray) and x_values.ndim == 1) \
        else x_values
    y_values = [y_values] if (isinstance(y_values, np.ndarray) and y_values.ndim == 1) \
        else y_values

    for x_vals, y_vals in zip(x_values, y_values):
        ax.plot(x_vals, y_vals, **plt_kwargs)

    _add_vlines(vline, ax)


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


@savefig
@set_plt_kwargs
def plot_polar_hist(data, bin_width=None, ax=None, **plt_kwargs):
    """Plot a polar histogram.

    Parameters
    ----------
    data : 1d array
        Data to plot in a circular histogram.
    bin_width : int, optional, default: 10
        Width of the bins to use for the histogram.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    if not ax:
        ax = plt.subplot(111, polar=True)

    bin_edges, counts = bin_circular(data, bin_width=bin_width)
    ax.bar(bin_edges[:-1], counts)


@savefig
@set_plt_kwargs
def plot_text(text, xpos=0.5, ypos=0.5, show_axis=False, ax=None, **plt_kwargs):
    """Plot text.

    Parameters
    ----------
    text : str
        The text to plot.
    xpos, ypos : float, optional, default: 0.5
        The x and y position to plot the text.
    show_axis : bool, optional, default: False
        Whether to show the axis of the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax)

    ax.text(xpos, ypos, text,
            fontdict=plt_kwargs.pop('fontdict', TEXT_SETTINGS['fontdict']),
            ha=plt_kwargs.pop('ha', TEXT_SETTINGS['ha']),
            va=plt_kwargs.pop('va', TEXT_SETTINGS['va']),
            **plt_kwargs)

    if not show_axis:
        ax.axis('off')
