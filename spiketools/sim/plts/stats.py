from spiketools.plts.utils import check_ax
from spiketools.plts.style import set_plt_kwargs

###################################################################################################
###################################################################################################

@set_plt_kwargs
def plot_param_stats(param_vals, stats, ax=None, **plt_kwargs):
    """Plot the statistics as a function of a parameter.

    Parameters
    ----------
    param_vals : array-like
        The parameter values.
    stats : array-like
        The statistics to plot.
    ax : Axes, optional
        The Matplotlib axis object to plot on. If not provided, a new axis will be created.
    plt_kwargs : dict
        Additional plotting arguments to be passed to the main plot functions.  
    """
    
    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))
    ax.plot(param_vals, stats, **plt_kwargs)
