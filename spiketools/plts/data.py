"""Plots for various data."""

from spiketools.utils.select import get_avg_func

from spiketools.plts.utils import check_ax, savefig, set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_hist(data, average=None, ax=None, **plt_kwargs):
    """Plot data as a histogram.

    Parameters
    ----------
    data : 1d array
        Data to plot.
    average : {'median', 'mean'}, optional
        Which kind of average to compute and add to the plot.
        If None, no average is plotted.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.

    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    ax.hist(data, **plt_kwargs)

    if average:
        ax.axvline(get_avg_func(average)(data), lw=4, color='red', alpha=0.8)


@savefig
@set_plt_kwargs
def plot_bar(data, labels=None, ax=None, **plt_kwargs):
    """Plot data in a bar graph.

    Parameters
    ----------
    data : list of float
        Data to plot.
    labels : list of str
        Labels for the bar plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    if not labels:
        labels = ['d' + str(ind) for ind in range(len(data))]

    ax.bar(labels, data, **plt_kwargs)
    ax.set(xlim=[-0.5, len(data)-0.5])
