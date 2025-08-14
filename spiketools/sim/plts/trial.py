import numpy as np
import matplotlib.pyplot as plt

from spiketools.plts.utils import check_ax, get_kwargs
from spiketools.plts.style import set_plt_kwargs
from spiketools.plts.data import  plot_lines
from spiketools.utils.options import get_avg_func, get_var_func

###################################################################################################
###################################################################################################

@set_plt_kwargs
def plot_trial_placefield(trial_placefield, spatial_bins=None, average=None, shade=None, add_traces=False,
                          trace_cmap=None, ax=None, **plt_kwargs):
    """
    Plot a trial placefield 

    Parameters
    ----------
    trial_placefield : 1D or 2D array
        The data representing trial placefields.
        If 2D, the array should have shape [n_trials, n_samples], with each row representing a trial.
        
    spatial_bins : 1D array, optional
        Binning or x-axis values corresponding to the trial placefield data. If not provided, indices are used.
        
    average : {'mean', 'median'}, optional
        Specifies averaging method to apply across trials if `trial_placefield` is a 2D array.
        When set, calculates and plots the average across all trials.
        
    shade : {'sem', 'std'} or 1D array, optional
        Variance measure to compute and plot as shaded regions around the average, or directly as a provided array.
        
    add_traces : bool, optional, default: False
        If True and `trial_placefield` is a 2D array, individual trial traces will be plotted on top of any averaged plot.
        
    cmap : str, optional
        Colormap name to use for individual traces if `add_traces` is True.
        
    ax : Axes, optional
        The Matplotlib axis object to plot on. If not provided, a new axis will be created.
        
    plt_kwargs : dict
        Additional plotting arguments to be passed to the main plot functions.
        
        Custom kwargs include:
        - 'traces_lw': Line width for individual traces.
        - 'traces_alpha': Transparency level for individual traces.
        - 'shade_alpha': Transparency level for the shaded region."""

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))

    custom_kwargs = ['traces_lw', 'traces_alpha', 'shade_alpha', 'shade_color', 'traces_color']
    custom_plt_kwargs = get_kwargs(plt_kwargs, custom_kwargs)
    
    all_trial_placefield = trial_placefield
    if isinstance(shade, str):
        shade = get_var_func(shade)(trial_placefield, 0)
        
    if isinstance(average, str):
        trial_placefield = get_avg_func(average)(trial_placefield, 0)
        

    xlabel = 'Time (s)'
    if spatial_bins is None:
        spatial_bins = np.arange(trial_placefield.shape[-1])
        xlabel = 'Spatial Bins'

    plot_lines(spatial_bins, trial_placefield, ax=ax,
               xlabel=plt_kwargs.pop('xlabel', xlabel),
               ylabel=plt_kwargs.pop('ylabel', 'Firing Rate (Hz)'),
               #title=plt_kwargs.pop('title', 'Firing Rate'),
               **plt_kwargs)

    if add_traces:
        if trace_cmap is not None:
            colormap = plt.get_cmap(trace_cmap)
            num_traces = all_trial_placefield.shape[0]
            
            for i in range(num_traces):
                color = colormap(i / num_traces)  # Normalize and get a color from the colormap
                ax.plot(
                    spatial_bins, 
                    all_trial_placefield[i].T,
                    lw=custom_plt_kwargs.pop('traces_lw', 1),
                    alpha=custom_plt_kwargs.pop('traces_alpha', 0.5),
                    color=color )
        else:
            ax.plot(
                spatial_bins, 
                all_trial_placefield.T,
                lw=custom_plt_kwargs.pop('traces_lw', 1),
                alpha=custom_plt_kwargs.pop('traces_alpha', 0.5),
                color=custom_plt_kwargs.pop('traces_color', ax.lines[0].get_color())
            )

    if shade is not None:
        ax.fill_between(spatial_bins, trial_placefield - shade, trial_placefield + shade,
                        alpha=custom_plt_kwargs.pop('shade_alpha', 0.25),
                        color=custom_plt_kwargs.pop('shade_color',ax.lines[0].get_color()))
