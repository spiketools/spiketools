"""Plots for tasks and task structure related visualizations."""

import matplotlib.pyplot as plt

from spiketools.plts.annotate import _add_shade, _add_vlines
from spiketools.plts.utils import check_ax, savefig, set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_task_structure(shades=None, lines=None, shade_colors=None, line_colors=None,
                        shade_kwargs=None, line_kwargs=None, ax=None):
    """Plot task structure with shaded regions and line events.

    Parameters
    ----------
    shades : list of list of float
        List of start and end ranges to shade in, to indicate event durations.
        To add multiple different shaded regions, pass a list of multiple shade definitions.
    lines : list of float
        Positions to draw vertical lines, to indicate point events.
        To add multiple different lines, pass a list of multiple line definitions.
    shade_colors : list of str
        Colors to plot the shades in. Used if passing multiple shade sections.
    line_colors : list of str
        Colors to plot the lines in. Used if passing multiple line sections.
    shade_kwargs : dict, optional
        Additional keyword arguments for the shades.
    line_kwargs : dict, optional
        Additional keyword arguments for the lines.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=(16, 2))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    shade_kwargs = {} if shade_kwargs is None else shade_kwargs
    line_kwargs = {} if line_kwargs is None else line_kwargs

    if shades is not None:
        if not isinstance(shades[0][0], (int, float)):
            for shade, color in zip(shades, shade_colors if shade_colors else color_cycle):
                shade_kwargs['color'] = color
                plot_task_structure(shades=shade, shade_kwargs=shade_kwargs, ax=ax)
        else:
            for st, en in zip(*shades):
                shade_kwargs.setdefault('alpha', 0.25)
                _add_shade([st, en], **shade_kwargs, ax=ax)

    if lines is not None:
        if not isinstance(lines[0], (int, float)):
            for line, color in zip(lines, line_colors if line_colors else color_cycle):
                line_kwargs['color'] = color
                plot_task_structure(lines=line, line_kwargs=line_kwargs, ax=ax)
        else:
            _add_vlines(lines, **line_kwargs, ax=ax)

    plt.yticks([])
