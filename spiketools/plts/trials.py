"""Plots for trials related measures and analyses."""

from spiketools.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
def plot_trial_rasters(data, line=0, xlim=None, show_axis=False, ax=None, **plt_kwargs):
    """Plot rasters across multiple trials.

    Parameters
    ----------
    trial_spikes : list of list of float
        Spike times per trial.
    line : float, optional, default: 0
        Position to draw a vertical line. If None, no line is drawn.
    xlim : list of float, optional
        Plot limits for the x-axis.
    show_axis : bool, optional, default: False
        Whether to show the axis around the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    ax.eventplot(data)

    if line is not None:
        ax.vlines(line, -1, len(data), lw=2.5, color='green', alpha=0.5);

    ax.set_xlim(xlim)

    ax.set(**plt_kwargs)

    if not show_axis:
        ax.set_axis_off();
