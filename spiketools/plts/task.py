"""Plots for tasks and task structure related visualizations."""

import numpy as np

from spiketools.utils.base import listify
from spiketools.plts.annotate import add_vshades, add_vlines
from spiketools.plts.utils import check_ax, savefig
from spiketools.plts.style import set_plt_kwargs
from spiketools.plts.settings import DEFAULT_COLORS

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_task_structure(task_ranges=None, event_lines=None, data_points=None,
                        range_colors=None, line_colors=None, range_kwargs=None, line_kwargs=None,
                        ax=None, **plt_kwargs):
    """Plot task structure, shaded ranges of event durations, and lines of point events.

    Parameters
    ----------
    task_ranges : list of list of float, optional
        List of start and end ranges to shade in, to indicate event durations.
        To add multiple different shaded regions, pass a list of multiple shade definitions.
    event_lines : list of float, optional
        Timestamps at which to draw vertical lines, to indicate point events.
        To add multiple different lines, pass a list of multiple line definitions.
    data_points : 1d array, optional
        Set of timestamps to indicate data points of interest on the plot.
    range_colors : list of str, optional
        Colors to plot the ranges in. Used if passing multiple task range sections.
    line_colors : list of str, optional
        Colors to plot the lines in. Used if passing multiple line sections.
    range_kwargs : dict, optional
        Additional keyword arguments for the range shades.
    line_kwargs : dict, optional
        Additional keyword arguments for the event lines.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', (16, 2)))

    range_kwargs = {} if range_kwargs is None else range_kwargs
    line_kwargs = {} if line_kwargs is None else line_kwargs

    if task_ranges is not None:
        if not isinstance(task_ranges[0][0], (int, float, np.int64, np.float64)):
            for trange, color in zip(task_ranges, range_colors if range_colors else DEFAULT_COLORS):
                range_kwargs['color'] = color
                plot_task_structure(task_ranges=trange, range_kwargs=range_kwargs, ax=ax)
        else:
            range_kwargs.setdefault('alpha', 0.25)
            shade_ranges = [[st, en] for st, en in zip(*task_ranges)]
            add_vshades(shade_ranges, **range_kwargs, ax=ax)

    if event_lines is not None:
        if not isinstance(event_lines[0], (int, float, np.int64, np.float64)):
            for eline, color in zip(event_lines, line_colors if line_colors else DEFAULT_COLORS):
                line_kwargs['color'] = color
                plot_task_structure(event_lines=list(eline), line_kwargs=line_kwargs, ax=ax)
        else:
            add_vlines(event_lines, **line_kwargs, ax=ax)

    if data_points is not None:
        ax.eventplot(data_points)

    ax.set_yticks([])


@savefig
@set_plt_kwargs
def plot_task_events(events, ax=None, **plt_kwargs):
    """Plot task events.

    Parameters
    ----------
    events : array or dict or list
        If array, indicates the times to plot event markers.
        If dict, should have the event markers as a 'times' key, plus any additional plot arguments.
        If list, can be a list of array or of dictionaries.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', (16, 2)))

    events = listify(events)
    for event in events:
        if isinstance(event, (list, np.ndarray)):
            ax.eventplot(event, **plt_kwargs)
        elif isinstance(event, dict):
            ax.eventplot(event.pop('times'), **event, **plt_kwargs)
