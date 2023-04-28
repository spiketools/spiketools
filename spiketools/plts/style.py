"""Functionality for styling plots."""

from functools import wraps

import matplotlib.pyplot as plt

from spiketools.utils.base import listify
from spiketools.utils.checks import check_param_options
from spiketools.plts.utils import check_ax, get_kwargs, get_attr_kwargs
from spiketools.plts.settings import SET_KWARGS, OTHER_KWARGS

###################################################################################################
###################################################################################################

def set_plt_kwargs(func):
    """Collects and then sets plot kwargs that can be applied with 'set'."""

    @wraps(func)
    def decorated(*args, **kwargs):

        setters = get_kwargs(kwargs, SET_KWARGS)
        title_kwargs = get_attr_kwargs(kwargs, 'title')

        others = get_kwargs(kwargs, OTHER_KWARGS)
        legend_kwargs = get_attr_kwargs(kwargs, 'legend')

        func(*args, **kwargs)

        ax = kwargs['ax'] if 'ax' in kwargs and kwargs['ax'] is not None else plt.gca()

        if 'title' in setters:
            ax.set_title(setters.pop('title'), **title_kwargs)

        ax.set(**setters)

        if 'legend' in others:
            ax.legend(others.pop('legend'), **legend_kwargs)

    return decorated


def drop_spines(sides, ax=None):
    """Drop spines from a plot axis.

    Parameters
    ----------
    sides : {'left', 'right', 'top', 'bottom'} or list
        Side(s) to drop spines from.
    ax : Axes, optional
        Axis object to update.
        If not provided, takes the current axis.
    """

    ax = check_ax(ax, return_current=True)

    for side in listify(sides):
        check_param_options(side, 'side', ['left', 'right', 'top', 'bottom'])
        ax.spines[side].set_visible(False)


def invert_axes(invert, ax=None):
    """Invert plot axes.

    Parameters
    ----------
    invert : {'x', 'y', 'both'} or None
        How to invert the plot axes, inverting the x, y, or both axes.
    ax : Axes, optional
        Axis object to update.
        If not provided, takes the current axis.

    Notes
    -----
    Note that for a 2d array, inverting axes is equivalent to flipping the data, specifically:

    - Flipping up/down is equivalent to inverting the y-axis.
    - Flipping left/right is equivalent to inverting the x-axis.
    """

    ax = check_ax(ax, return_current=True)

    if invert in ['x', 'both']:
        ax.invert_xaxis()
    if invert in ['y', 'both']:
        ax.invert_yaxis()


def add_gridlines(x_bins, y_bins, ax):
    """Add gridlines to a plot axis.

    Parameters
    ----------
    x_bins, y_bins : list of float, optional
        Bin edges for each axis.
        If provided, these are used to draw grid lines on the plot.
    ax : Axes, optional
        Axis object to update.
        If not provided, takes the current axis.
    """

    ax = check_ax(ax, return_current=True)

    ax.set_xticks(x_bins if x_bins is not None else [], minor=False)
    ax.set_yticks(y_bins if y_bins is not None else [], minor=False)

    # Although this looks like it doubles the above code, what actually happens
    #   is that the above sets the ticks, and the below removes them from being displayed
    #   but they are still there, meaning grid lines get added when the grid call added
    # Note: it's tricky to have different grid lines and axis labels
    #   To refactor to do this, could use axvline / axhline to add custom lines
    ax.set(xticklabels=[], yticklabels=[])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    if x_bins is not None or y_bins is not None:
        ax.grid()
