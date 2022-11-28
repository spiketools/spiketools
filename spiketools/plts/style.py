"""Functionality for styling plots."""

from functools import wraps

import matplotlib.pyplot as plt

from spiketools.utils.checks import check_list_options
from spiketools.plts.settings import SET_KWARGS, OTHER_KWARGS

###################################################################################################
###################################################################################################

def get_kwargs(kwargs, select):
    """Get keyword arguments.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to extract from.
    select : list of str
        The arguments to extract.

    Returns
    -------
    setters : dict
        Selected keyword arguments.
    """

    setters = {arg : kwargs.pop(arg, None) for arg in select}
    setters = {arg : value for arg, value in setters.items() if value is not None}

    return setters


def get_attr_kwargs(kwargs, attr):
    """Get keyword arguments related to a particular attribute.

    Parameters
    ----------
    kwargs : dict
        Plotting related keyword arguments.
    attr : str
        The attribute to select related arguments.

    Returns
    -------
    attr_kwargs : dict
        Selected keyword arguments, related to the given attribute.
    """

    labels = [key for key in kwargs.keys() if attr in key]
    attr_kwargs = {label.split('_')[1] : kwargs.pop(label) for label in labels}

    return attr_kwargs


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


def drop_spines(ax, sides):
    """Drop spines from a plot axis.

    Parameters
    ----------
    ax : Axes
        Axis object to update.
    sides : {'left', 'right', 'top', 'bottom'} or list
        Side(s) to drop spines from.
    """

    sides = [sides] if isinstance(sides, str) else sides
    check_list_options(sides, 'sides', ['left', 'right', 'top', 'bottom'])
    for side in sides:
        ax.spines[side].set_visible(False)
