"""Plot cell place field data with a colormap visualization."""

import matplotlib.pyplot as plt
from spiketools.plts.utils import check_ax, savefig
from spiketools.plts.style import set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_neuron_placefield(vals, neuron_place_bins,
                          colormap_name='Greys', ax=None, **plt_kwargs):
    """Plot cell place field data with a colormap visualization.

    Parameters
    ----------
    vals: array-like
        Values to be plotted.
    neuron_place_bins: array-like
        Cell place bins to be plotted.
    colormap_name: str
        Colormap name to be used.
    ax: matplotlib.axes.Axes
        Axes object.
    plt_kwargs: dict
        Additional plotting arguments to be passed to the main plot functions.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    colormap = plt.get_cmap(colormap_name)
    for i in range(len(vals)):
        for j in range(len(neuron_place_bins[0])):
            color = colormap(i / len(vals))
            ax.plot(neuron_place_bins[i][j], color=color, **plt_kwargs)
