"""Helper functions to annotate plots with extra elements / information."""

from itertools import repeat

from spiketools.plts.utils import check_ax

###################################################################################################
###################################################################################################

def color_pval(p_value, alpha=0.05, significant_color='red', null_color='black'):
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


def _add_vlines(vline, ax=None, **plt_kwargs):
    """Add vertical line(s) to a plot axis.

    Parameters
    ----------
    vline : float or list
        Positions(s) of the vertical lines to add to the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if vline is not None:
        vline = [vline] if isinstance(vline, (int, float)) else vline
        for line in vline:
            ax.axvline(line, **plt_kwargs)


def _add_hlines(hline, ax=None, **plt_kwargs):
    """Add horizontal line(s) to a plot axis.

    Parameters
    ----------
    hline : float or list
        Positions(s) of the horizontal lines to add to the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if hline is not None:
        hline = [hline] if isinstance(hline, (int, float)) else hline
        for line in hline:
            ax.axhline(line, **plt_kwargs)


def _add_vshade(vshade, ax=None, **plt_kwargs):
    """Add vertical shading to a plot axis.

    Parameters
    ----------
    vshade : list of float
        Region of the plot to shade in.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if vshade is not None:
        ax.axvspan(*vshade, **plt_kwargs)


def _add_hshade(hshade, ax=None, **plt_kwargs):
    """Add horizontal shading to a plot axis.

    Parameters
    ----------
    hshade : list of float
        Region of the plot to shade in.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if hshade is not None:
        ax.axhspan(*hshade, **plt_kwargs)


def _add_box_shade(x1, x2, y_val, y_range=0.41, ax=None, **plt_kwargs):
    """Add a shaded box to a plot axis.

    Parameters
    ----------
    x1, x2 : float
        The start and end positions for the shaded box on the x-axis.
    y_val : float
        The position for the shaded box on the y-axis.
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


def _add_box_shades(x_values, y_values=None, x_range=1, y_range=0.41, ax=None, **plt_kwargs):
    """Add multiple shaded boxes to a plot axis.

    Parameters
    ----------
    x_values, y_values : 1d array
        A list of center position values for the x- and y-axes for each shaded box.
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
        _add_box_shade(xval - x_range, xval + x_range, yval, y_range,
                       color=color, ax=ax, **plt_kwargs)


def _add_dots(dots, ax=None, **plt_kwargs):
    """Add dots to a plot axis.

    Parameters
    ----------
    dots : 2d array
        Definitions of the dots to add to the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, return_current=True)

    if dots is not None:
        ax.plot(dots[0, :], dots[1, :], linestyle='',
                marker=plt_kwargs.pop('marker', '.'),
                **plt_kwargs)


def _add_significance(stats, sig_level=0.05, x_vals=None, ax=None):
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


def _add_side_text(texts, colors, y_values=None, ax=None, **plt_kwargs):
    """Add text to the side of a plot.

    Parameters
    ----------
    texts : list of str
        Text(s) to add to the plot.
    colors : str or list of str
        Color(s) for each entry.
    y_values : list of float, optional
        xx
    ax : Axes, optional
        Axis object upon which to plot.
    """

    ax = check_ax(ax, return_current=True)

    if not y_values:
        y_values = range(len(texts))

    colors = repeat(colors) if isinstance(colors, str) else colors

    plot_end = ax.get_xlim()[1]

    for text, color, yval in zip(texts, colors, y_values):
        ax.text(plot_end + 5, yval, text, color=color,
                fontsize=plt_kwargs.pop('fontsize', 11),
                fontweight=plt_kwargs.pop('fontweight', 'bold'),
                **plt_kwargs)
