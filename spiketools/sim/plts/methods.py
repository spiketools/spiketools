
import matplotlib.pyplot as plt

from spiketools.plts.utils import check_ax
from spiketools.plts.style import set_plt_kwargs

###################################################################################################
###################################################################################################

@set_plt_kwargs
def method_method_comparison(method_A, method_B, param_vals, color_map, cbar = True, ax=None,cbar_kwargs=None, vmin=None, vmax=None,    **plt_kwargs):

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))
    scatter = ax.scatter(method_A, method_B, c=param_vals, cmap= color_map, marker='*', s=200,vmin=vmin, vmax=vmax  )
    if cbar:
        cbar_kwargs = cbar_kwargs or {}
        cbar = plt.colorbar(scatter, ax=ax)
        if 'ylabel' in cbar_kwargs:
            cbar.ax.set_ylabel(cbar_kwargs['ylabel'])
