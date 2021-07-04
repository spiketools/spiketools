"""Plots for statistical analyses and related visualizations."""

from spiketools.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
def plot_surrogates(surrogates, data_value=None, p_value=None, figsize=None, ax=None):
    """Plot a distribution of surrogate data."""

    ax = check_ax(ax, figsize=figsize)

    ax.hist(surrogates)

    if data_value is not None:
        ax.axvline(data_value, color='k', linestyle='dashed', linewidth=2)

    if p_value is not None:
        ax.text(0.15, 0.9, 'p={:4.4f}'.format(p_value),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
