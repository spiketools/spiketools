"""Plots for spikes."""

import seaborn as sns

from spiketools.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
def plot_waveform(waveform, ax=None):
    """Plot a spike waveform.

    Parameters
    ----------
    waveform : 1d array
        Voltage values of the spike waveform.
    """

    ax = check_ax(ax)

    ax.plot(waveform)
    ax.set(title='Spike Waveform')


@savefig
def plot_isis(isis, bins=None, range=None, density=False, ax=None):
    """Plot a distribution of ISIs.

    Parameters
    ----------
    isis : 1d array
        Interspike intervals.
    """

    ax = check_ax(ax)

    ax.hist(isis, bins=bins, range=range, density=density);
    ax.set(xlabel='Time', title='ISIs')


@savefig
def plot_firing_rates(rates, figsize=None, ax=None):
    """Plot a bar plot of firing rates for a group of neurons.

    Parameters
    ----------
    rates : list of float
        Firing rates.
    """

    ax = check_ax(ax, figsize)

    labels = ['U' + str(ind) for ind in range(len(rates))]
    sns.barplot(x=labels, y=rates, ax=ax)

    ax.set(xlabel='Units', ylabel='Firing Rate (Hz)', title='Firing Rates for all Units')
