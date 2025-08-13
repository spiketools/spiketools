from spiketools.plts.utils import check_ax, get_kwargs, savefig
from spiketools.plts.style import set_plt_kwargs

###################################################################################################
###################################################################################################

@savefig
@set_plt_kwargs
def plot_firing_rate(data,ax=None,**plt_kwargs):
    
    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))
    
    custom_kwargs = ['data_value_color', 'data_value_linestyle', 'data_value_lw']
    custom_plt_kwargs = get_kwargs(plt_kwargs, custom_kwargs)
    
    ax.plot(data, **custom_plt_kwargs)


    
