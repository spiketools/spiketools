"""Plots for spikes."""

import numpy as np
import matplotlib.pyplot as plt

from spiketools.utils.options import get_avg_func, get_var_func
from spiketools.plts.data import plot_bar, plot_hist, plot_lines
from spiketools.plts.utils import check_ax, get_kwargs, savefig
from spiketools.plts.style import set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_waveform(waveform, timestamps=None, average=None, shade=None, add_traces=False,
                  ax=None, **plt_kwargs):
    """Plot a spike waveform.

    Parameters
    ----------
    waveform : 1d or 2d array
        Voltage values of the spike waveform(s).
        If 2d, should have shape [n_waveforms, n_timestamps].
    timestamps : 1d array, optional
        Timestamps corresponding to the waveform(s).
    average : {'mean', 'median'}, optional
        Averaging to apply to waveforms before plotting.
        If provided, this takes an average across an assumed 2d array of waveforms.
    shade : {'sem', 'std'} or 1d array, optional
        Measure of variance to compute and/or plot as shading.
    add_traces : bool, optional, default: False
        Whether to also plot individual waveform traces.
        Only applicable if `waveform` is a 2d array.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
        Custom kwargs: 'traces_lw', 'traces_alpha', 'shade_alpha'.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    custom_kwargs = ['traces_lw', 'traces_alpha', 'shade_alpha']
    custom_plt_kwargs = get_kwargs(plt_kwargs, custom_kwargs)

    if isinstance(shade, str):
        shade = get_var_func(shade)(waveform, 0)

    if isinstance(average, str):
        all_waveforms = waveform
        waveform = get_avg_func(average)(waveform, 0)

    xlabel = 'Time (s)'
    if timestamps is None:
        timestamps = np.arange(waveform.shape[-1])
        xlabel = 'Samples'

    plot_lines(timestamps, waveform, ax=ax,
               xlabel=plt_kwargs.pop('xlabel', xlabel),
               ylabel=plt_kwargs.pop('ylabel', 'Voltage'),
               title=plt_kwargs.pop('title', 'Spike Waveform'),
               **plt_kwargs)

    if add_traces:
        ax.plot(timestamps, all_waveforms.T,
                lw=custom_plt_kwargs.pop('traces_lw', 1),
                alpha=custom_plt_kwargs.pop('traces_alpha', 0.5),
                color=ax.lines[0].get_color())

    if shade is not None:
        ax.fill_between(timestamps, waveform - shade, waveform + shade,
                        alpha=custom_plt_kwargs.pop('shade_alpha', 0.25))


@savefig
@set_plt_kwargs
def plot_waveforms3d(waveforms, timestamps=None, **plt_kwargs):
    """Plot waveforms on a 3D axis.

    Parameters
    ----------
    waveforms : 2d array
        Voltage values for the waveforms, with shape [n_waveforms, n_timestamps].
    timestamps : 1d array, optional
        Timestamps corresponding to the waveforms.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    plt.figure(figsize=plt_kwargs.pop('figsize', None))
    ax = plt.subplot(projection='3d')

    if timestamps is None:
        timestamps = np.arange(waveforms.shape[-1])

    ys = np.ones(waveforms.shape[1])
    for ind, waveform in enumerate(waveforms):
        ax.plot(timestamps, ys * ind, waveform)

    # Set axis view orientation and hide axes
    ax.view_init(None, None)
    ax.axis('off')
    ax.set(title=plt_kwargs.pop('title', 'Spike Waveforms'))


@savefig
@set_plt_kwargs
def plot_waveform_density(waveforms, timestamps=None, bins=(250, 50), cmap='viridis',
                          ax=None, **plt_kwargs):
    """Plot a heatmap of waveform density, created as a 2d histogram of spike waveforms.

    Parameters
    ----------
    waveforms : 2d array
        Voltage values for the waveforms, with shape [n_waveforms, n_timestamps].
    timestamps : 1d array, optional
        Timestamps corresponding to the waveforms.
    bins : tuple of (int, int), optional, default: (250, 50)
        Bin definition to use to create the figure.
    cmap : str, optional, default: 'viridis'
        Colormap to use for the figure.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    xlabel = 'Time (s)'
    if timestamps is None:
        timestamps = np.arange(waveforms.shape[-1])
        xlabel = 'Samples'
    timestamps = np.vstack([timestamps] * waveforms.shape[0])

    ax.hist2d(timestamps.flatten(), waveforms.flatten(), bins=bins, cmap=cmap)

    ax.set(xlabel=plt_kwargs.pop('xlabel', xlabel),
           ylabel=plt_kwargs.pop('ylabel', 'Voltage'),
           title=plt_kwargs.pop('title', 'Spike Histogram'))


@savefig
@set_plt_kwargs
def plot_isis(isis, bins=None, range=None, density=False, ax=None, **plt_kwargs):
    """Plot a distribution of ISIs.

    Parameters
    ----------
    isis : 1d array
        Interspike intervals.
    bins : int or list, optional
        Bin definition, either a number of bins to use, or bin definitions.
    range : tuple, optional
        Range of the data to plot.
    density : bool, optional, default: False
        Whether to draw a probability density.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    plot_hist(isis, bins, range, density, ax=ax,
              xlabel=plt_kwargs.pop('xlabel', 'Time'),
              title=plt_kwargs.pop('title', 'ISIs'),
              **plt_kwargs)


@savefig
@set_plt_kwargs
def plot_firing_rates(rates, ax=None, **plt_kwargs):
    """Plot firing rates for a group of neurons.

    Parameters
    ----------
    rates : list of float
        Firing rates.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    plot_bar(rates, labels=['U' + str(ind) for ind in range(len(rates))], ax=ax,
             xlabel=plt_kwargs.pop('xlabel', 'Units'),
             ylabel=plt_kwargs.pop('xlabel', 'Firing Rate (Hz)'),
             title=plt_kwargs.pop('title', 'Firing Rates of all Units'),
             **plt_kwargs)
