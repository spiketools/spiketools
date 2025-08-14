
import matplotlib.pyplot as plt

from spiketools.plts.utils import check_ax
from spiketools.plts.style import set_plt_kwargs

###################################################################################################
###################################################################################################

@set_plt_kwargs
def method_method_comparison(method_A, method_B, param_vals, color_map, cbar = True, ax=None, cbar_kwargs=None, vmin=None, vmax=None, **plt_kwargs):
    """Plot a comparison of two methods.    
    
    Parameters
    ----------
    method_A: array-like
        Method A data
    method_B: array-like
        Method B data
    param_vals: array-like
        Parameter values
    color_map: str
        Color map
    cbar: bool
        Whether to add a color bar
    ax: matplotlib.axes.Axes
        Axes object
    cbar_kwargs: dict
        Color bar kwargs
    vmin: float
        Minimum value for the color map
    vmax: float
        Maximum value for the color map
    plt_kwargs: dict
        Additional plotting arguments to be passed to the main plot functions.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))
    scatter = ax.scatter(method_A, method_B, c=param_vals, cmap= color_map, marker='*', s=200,vmin=vmin, vmax=vmax  )
    if cbar:
        cbar_kwargs = cbar_kwargs or {}
        cbar = plt.colorbar(scatter, ax=ax)
        if 'ylabel' in cbar_kwargs:
            cbar.ax.set_ylabel(cbar_kwargs['ylabel'])
