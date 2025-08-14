"""Plots for statistical analyses and related visualizations."""

from spiketools.plts.data import plot_hist
from spiketools.plts.utils import check_ax, get_kwargs, savefig
from spiketools.plts.style import set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_surrogates(surrogates, data_value=None, p_value=None, ax=None, **plt_kwargs):
    """Plot a distribution of surrogate data.

    Parameters
    ----------
    surrogates : 1d array
        The collection of values computed on surrogates.
    data_value : float, optional
        The statistic value of the real data, to draw on the plot.
    p_value : float, optional
        The p-value to print on the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
        Custom kwargs: 'data_value_color', 'data_value_linestyle', 'data_value_lw'.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    custom_kwargs = ['data_value_color', 'data_value_linestyle', 'data_value_lw']
    custom_plt_kwargs = get_kwargs(plt_kwargs, custom_kwargs)

    plot_hist(surrogates, ax=ax, **plt_kwargs)

    if data_value is not None:
        ax.axvline(data_value,
                   color=custom_plt_kwargs.pop('data_value_color', 'k'),
                   linestyle=custom_plt_kwargs.pop('data_value_linestyle', 'dashed'),
                   lw=custom_plt_kwargs.pop('data_value_lw', 2))

    if p_value is not None:
        ax.text(0.15, 0.9, 'p={:4.4f}'.format(p_value),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
