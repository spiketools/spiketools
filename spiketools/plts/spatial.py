"""Plots for spatial measures and analyses."""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from spiketools.utils.data import smooth_data, compute_range
from spiketools.modutils.functions import get_function_parameters
from spiketools.plts.annotate import _add_dots
from spiketools.plts.settings import DEFAULT_COLORS
from spiketools.plts.utils import check_ax, savefig, set_plt_kwargs, make_axes

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_positions(positions, spike_positions=None, landmarks=None,
                   x_bins=None, y_bins=None, ax=None, **plt_kwargs):
    """Plot positions.

    Parameters
    ----------
    positions : 2d array or list of 2d array
        Position data.
        If a list, each element of the list is plotted separately, on the same plot.
    spike_positions : 2d array or dict, optional
        Position values of spikes, to indicate on the plot.
        If array, defines the positions.
        If dictionary, should include a 'positions' key plus additional plot arguments.
    landmarks : 2d array or dict or list, optional
        Position values of landmarks, to be added to the plot.
        If array, defines the positions.
        If dictionary, should include a 'positions' key plus additional plot arguments.
        Multiple landmarks can be added by passing a list of arrays or a list of dictionaries.
    x_bins, y_bins : list of float, optional
        Bin edges for each axis.
        If provided, these are used to draw grid lines on the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    positions = [positions] if isinstance(positions, np.ndarray) else positions
    for cur_positions in positions:
        ax.plot(*cur_positions,
                color=plt_kwargs.pop('color', DEFAULT_COLORS[0]),
                alpha=plt_kwargs.pop('alpha', 0.35),
                **plt_kwargs)

    if spike_positions is not None:
        defaults = {'color' : 'red', 'alpha' : 0.4, 'ms' : 6}
        if isinstance(spike_positions, np.ndarray):
            _add_dots(spike_positions, ax=ax, **defaults)
        elif isinstance(spike_positions, dict):
            _add_dots(spike_positions.pop('positions'), ax=ax, **{**defaults, **spike_positions})

    if landmarks is not None:
        defaults = defaults = {'alpha' : 0.85, 'ms' : 12}
        for landmark in [landmarks] if not isinstance(landmarks, list) else landmarks:
            if isinstance(landmark, np.ndarray):
                _add_dots(landmark, ax=ax, **defaults)
            elif isinstance(landmark, dict):
                _add_dots(landmark.pop('positions'), ax=ax, **landmark)

    if x_bins is not None:
        ax.set_xticks(x_bins, minor=False)
    if y_bins is not None:
        ax.set_yticks(y_bins, minor=False)

    ax.set(xticklabels=[], yticklabels=[])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    if x_bins is not None or y_bins is not None:
        ax.grid()


@savefig
@set_plt_kwargs
def plot_position_by_time(times, positions, spike_times=None, spike_positions=None,
                          ax=None, **plt_kwargs):
    """Plot the position across time for a single dimension.

    Parameters
    ----------
    times : 1d array
        Time values associated with the position values.
    positions : 1d array
        Position values, for a single dimension.
    spike_times : 1d array, optional
        Timepoints at which spikes occur.
    spike_positions : 1d array, optional
        Position values of spikes, to indicate on the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    spikes = None
    if spike_times is not None:
        spikes = np.array([spike_times, spike_positions])

    plot_positions(np.array([times, positions]), spikes, ax=ax, **plt_kwargs)

    ax.set(xlabel='Time', ylabel='Position')


@savefig
@set_plt_kwargs
def plot_heatmap(data, transpose=False, smooth=False, smoothing_kernel=1.5,
                 ignore_zero=False, cbar=False, cmap=None, vmin=None, vmax=None,
                 ax=None, **plt_kwargs):
    """Plot a spatial heat map.

    Parameters
    ----------
    data : 2d array
        Measure to plot across a grided environment.
    transpose : bool, optional, default: False
        Whether to transpose the data before plotting.
    smooth : bool, optional, default: False
        Whether to smooth the data before plotting.
    smoothing_kernel : float, optional, default: 1.5
        Standard deviation of the gaussian kernel to apply for smoothing.
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

    Notes
    -----
    This function uses `plt.imshow` to visualize the matrix.
    Note that in doing so, it defaults to settings the origin to 'lower'.
    This setting can be overwritten by passing in a value for `origin`.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    if data.ndim < 2:
        data = np.atleast_2d(data)

    if transpose:
        data = data.T

    if smooth:
        data = smooth_data(data, smoothing_kernel)

    if ignore_zero:
        data = deepcopy(data)
        data[data == 0.] = np.nan

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin=plt_kwargs.pop('origin', 'lower'),
                   **plt_kwargs)

    ax.set(xticks=[], yticks=[])
    ax.set_axis_off()

    if cbar:
        colorbar = plt.colorbar(im)
        colorbar.outline.set_visible(False)


@savefig
def plot_trial_heatmaps(trial_data, **plt_kwargs):
    """Plot spatial heat maps for a set of trials.

    Parameters
    ----------
    trial_data : 3d array
        Spatially binned spike activity, per trial, with shape of [n_trials, n_xbins, n_ybins].
    plt_kwargs
        Additional arguments to pass into the plot function.
        This can include argument into `make_axes`, which initialize the set of axes.
    """

    axis_kwargs = {key : plt_kwargs.pop(key) \
        for key in get_function_parameters(make_axes) if key in plt_kwargs}
    axes = make_axes(trial_data.shape[0], **axis_kwargs)
    for data, ax in zip(trial_data, axes):
        plot_heatmap(data, **plt_kwargs, ax=ax)


def create_heat_title(label, data, stat=None, p_val=None):
    """Create a standardized title for an heatmap, listing the data range.

    Parameters
    ----------
    label : str
        Label to add to the beginning of the title.
    data : 2d array
        The array of data that is plotted, used to compute the data range.
    stat, p_val : float, optional
        A statistical test value and p statistic to report related to the heatmap.

    Returns
    -------
    title : str
        Title for the plot.
    """

    template = '({:1.2f}-{:1.2f})' if 'float' in str(data.dtype) else '({:d}-{:d})'
    if stat is None:
        title = ('{} - ' + template).format(label, *compute_range(data))
    else:
        title = ('{} - ' + template + '\n stat: {:1.2f}, p: {:1.2f}').format(\
            label, *compute_range(data), stat, p_val)

    return title
