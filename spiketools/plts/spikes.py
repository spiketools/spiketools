"""Plots for spikes."""

from spiketools.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
def plot_waveform(waveform, ax=None, **plt_kwargs):
    """Plot a spike waveform.

    Parameters
    ----------
    waveform : 1d array
        Voltage values of the spike waveform.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    ax.plot(waveform, **plt_kwargs)
    ax.set(title='Spike Waveform')


@savefig
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

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    ax.hist(isis, bins=bins, range=range, density=density, **plt_kwargs)
    ax.set(xlabel='Time', title='ISIs')


@savefig
def plot_firing_rates(rates, ax=None):
    """Plot a bar plot of firing rates for a group of neurons.

    Parameters
    ----------
    rates : list of float
        Firing rates.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    labels = ['U' + str(ind) for ind in range(len(rates))]

    ax.bar(labels, rates, **plt_kwargs)

    ax.set(xlabel='Units',
           ylabel='Firing Rate (Hz)',
           title='Firing Rates for all Units')

    ax.set_xlim([-0.5, len(rates)-0.5])
