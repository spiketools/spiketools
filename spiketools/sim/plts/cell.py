
import numpy as np
import matplotlib.pyplot as plt

from spiketools.plts.utils import check_ax, get_kwargs,savefig
from spiketools.plts.style import set_plt_kwargs


@savefig
@set_plt_kwargs
def plot_cell_placefield(vals : np.ndarray, cell_place_bins : np.ndarray, colormap_name='Greys', ax=None,**plt_kwargs):
    """
    Plot cell place field data with a colormap visualization

    Parameters
    ----------
    vals: array-like
        Values to be plotted
    cell_place_bins: array-like
        Cell place bins to be plotted
    colormap_name: str
        Colormap name to be used
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))
    
    custom_kwargs = ['data_value_color', 'data_value_linestyle', 'data_value_lw']
    custom_plt_kwargs = get_kwargs(plt_kwargs, custom_kwargs)


    colormap = plt.get_cmap(colormap_name)
    for i in range(len(vals)):
        for j in range(len(cell_place_bins[0])):
            color = colormap(i / len(vals))
            ax.plot(cell_place_bins[i][j], color=color, **plt_kwargs)

            



