"""Helper functions to annotate plots with extra elements / information."""

from itertools import repeat

import numpy as np

from spiketools.utils.base import listify
from spiketools.utils.checks import check_param_options
from spiketools.plts.utils import check_ax

###################################################################################################
###################################################################################################

def color_pvalue(p_value, alpha=0.05, significant_color='red', null_color='black'):
    """Select a color based on the significance of a p-value.

    Parameters
    ----------
    p_value : float
        The p-value to check.
    alpha : float, optional, default: 0.05
        The significance level to check against.
    signicant_color : str, optional, default: 'red'
        The color for if the p-value is significant.
    null_color : str, optional, default: 'black'
        The color for if the p-value is not significant.

    Returns
    -------
    color : str
        Color value, reflecting the significance of the given p-value.
    """

    return significant_color if p_value < alpha else null_color


def add_vlines(vline, ax=None, **plt_kwargs):
    """Add vertical line(s) to a plot axis.

    Parameters
    ----------
    vline : float or list
        Location(s) of the vertical lines to add to the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if vline is not None:
        for line in listify(vline):
            ax.axvline(line, **plt_kwargs)


def add_hlines(hline, ax=None, **plt_kwargs):
    """Add horizontal line(s) to a plot axis.

    Parameters
    ----------
    hline : float or list
        Location(s) of the horizontal lines to add to the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if hline is not None:
        for line in listify(hline):
            ax.axhline(line, **plt_kwargs)


def add_gridlines(x_bins, y_bins, ax=None, **plt_kwargs):
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
        ax.grid(**plt_kwargs)


def add_vshades(vshades, ax=None, **plt_kwargs):
    """Add vertical shade region(s) to a plot axis.

    Parameters
    ----------
    vshade : list of float or list of list of float
        Region(s) of the plot to shade in.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if vshades is not None:
        for vshade in listify(vshades, index=True):
            ax.axvspan(*vshade, **plt_kwargs)


def add_hshades(hshades, ax=None, **plt_kwargs):
    """Add horizontal shade region(s) to a plot axis.

    Parameters
    ----------
    hshade : list of float
        Region(s) of the plot to shade in.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if hshades is not None:
        for hshade in listify(hshades, index=True):
            ax.axhspan(*hshade, **plt_kwargs)


def add_box_shade(x1, x2, y_val, y_range=0.41, ax=None, **plt_kwargs):
    """Add a shaded box to a plot axis.

    Parameters
    ----------
    x1, x2 : float
        The start and end positions for the shaded box on the x-axis.
    y_val : float
        The position of the shaded box on the y-axis.
    y_range : float
        The range, as +/-, around the y position to shade the box.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    ax.fill_between([x1, x2], [y_val+y_range, y_val+y_range], [y_val-y_range, y_val-y_range],
                    alpha=plt_kwargs.pop('alpha', 0.2), **plt_kwargs)


def add_box_shades(x_values, y_values=None, x_range=1, y_range=0.41, ax=None, **plt_kwargs):
    """Add multiple shaded boxes to a plot axis.

    Parameters
    ----------
    x_values, y_values : 1d array
        Center position values for the x- and y-axes for each shaded box.
    x_range, y_range : float
        The range, as +/-, around the x and y positions to shade the box.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    color = plt_kwargs.pop('color', 'blue')

    if y_values is None:
        y_values = range(0, len(x_values))

    for xval, yval in zip(x_values, y_values):
        add_box_shade(xval - x_range, xval + x_range, yval, y_range,
                      color=color, ax=ax, **plt_kwargs)


def add_dots(dots, ax=None, **plt_kwargs):
    """Add dots to a plot axis.

    Parameters
    ----------
    dots : 1d or 2d array
        Definitions of the dots to add to the plot.
        If 1d array, defines a single dot as [x_pos, y_pos].
        If 2d array, 0th row is x-pos and 1th row is y-pos for multiple dot positions.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if dots is not None:

        # If dots are 1d, convert to 2d, transposing to match row organization
        dots = np.atleast_2d(dots).T if dots.ndim == 1 else dots

        ax.plot(dots[0, :], dots[1, :], linestyle='',
                marker=plt_kwargs.pop('marker', '.'),
                **plt_kwargs)


def add_significance(stats, sig_level=0.05, x_vals=None, ax=None):
    """Add markers to a plot axis to indicate statistical significance.

    Parameters
    ----------
    stats : list
        Statistical results, including p-values, to use to annotate the plot.
        List can contain floats, or statistical results if it has a `pvalue` field.
    sig_level : float, optional, default: 0.05
        Threshold level to consider a result significant.
    x_vals : 1d array, optional
        Values for the x-axis, for example time values or bin numbers.
        If not provided, x-values are accessed from the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    """

    ax = check_ax(ax, return_current=True)

    if not x_vals:
        x_vals = ax.lines[0].get_xdata()

    if not isinstance(stats[0], (float)):
        stats = [stat.pvalue for stat in stats]

    for ind, stat in enumerate(stats):
        if stat < sig_level:
            ax.plot(x_vals[ind], 0, '*', color='black')


def add_text_labels(texts, location='start', axis='x', offset=None,
                    values=None, colors='black', ax=None, **plt_kwargs):
    """Add text to the side of a plot.

    Parameters
    ----------
    texts : list of str
        Text(s) to add to the plot.
    location : {'start', 'end'} or iterable
        Location to plot the text labels across the axis.
    axis : {'x', 'y'}
        Which axis to add text labels across.
    offset : float, optional
        An offset value to move the text.
        If not provided, default to 10% of the plot range.
    values : list of float, optional
        Position values to plot the text on the axis defined in `axis`.
        If not provided, defaults to the indices of the text labels.
    colors : str or list of str, optional
        Color(s) for each entry. Defaults to 'black'.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional keyword arguments to pass to the `plt.text` call.
    """

    ax = check_ax(ax, return_current=True)

    check_param_options(axis, 'axis', ['x', 'y'])

    plot_range = getattr(ax, 'get_' + {'x' : 'y', 'y' : 'x'}[axis] + 'lim')()

    if not offset:
        offset = 0.15 * np.max(plot_range)
    if not values:
        values = range(len(texts))

    colors = repeat(colors) if isinstance(colors, str) else colors

    if isinstance(location, str):
        check_param_options(location, 'location', ['start', 'end'])
        ind = {'start' : 0, 'end' : 1}[location]
        location = repeat(plot_range[ind])
    else:
        location = iter(location)
        offset = -offset

    for text, color, value in zip(texts, colors, values):
        if axis == 'x':
            tpos = [value, next(location) + offset]
        if axis == 'y':
            tpos = [next(location) + offset, value]
        ax.text(*tpos, text, color=color,
                fontsize=plt_kwargs.pop('fontsize', None),
                fontweight=plt_kwargs.pop('fontweight', 'bold'),
                ha='center', va='center',
                **plt_kwargs)
