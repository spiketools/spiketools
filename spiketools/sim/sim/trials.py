import numpy as np
import matplotlib.pyplot as plt

from spiketools.plts.utils import make_grid,get_grid_subplot
from spiketools.plts.style import drop_spines

from placefield import sim_placefield,sim_skew_placefield,sim_multi_placefield,sim_multi_skew_placefield

import sys
sys.path.append('../code/plts')
from trial import plot_trial_placefield

###################################################################################################
###################################################################################################

def sim_trial_placefield(height_mean, height_std, width_mean, width_std, place_loc_mean, place_loc_std, n_bins, noise_std, base_mean, base_std, n_trials, vary_height=True, vary_width=True, vary_place_loc=True, plot=True, presence_ratio=0.6):
    """Simulate multiple trials of place fields with optional variability in height, width, and location.
    
    Parameters
    ----------
    height_mean : float
        Mean peak firing rate for the place field.
    height_std : float
        Standard deviation for the peak firing rate.
    width_mean : float
        Mean width of the place field.
    width_std : float
        Standard deviation for the width of the place field.
    place_loc_mean : float
        Mean location of the center of the place field.
    place_loc_std : float
        Standard deviation for the location of the center of the place field.
    n_bins : int
        Number of spatial bins over which the place field is simulated.
    noise_std : float
        Standard deviation for the noise added to the place field.
    base_mean : float
        Mean baseline firing rate.
    base_std : float
        Standard deviation of the baseline firing rate. 
    n_trials : int
        Number of trials to simulate. 
    vary_height : bool, optional
        If True, the height of the place field varies across trials. Default is True.
    vary_width : bool, optional
        If True, the width of the place field varies across trials. Default is True.
    vary_place_loc : bool, optional
        If True, the location of the place field center varies across trials. Default is True.
    plot : bool, optional
        If True, plots the individual trials and their average. Default is True. 
    presence_ratio : float, optional
        The fraction of trials to simulate with data, with the remainder being filled with zeros. Default is 0.6.

    Returns
    -------
    trial_place_bins : 2D array
        Simulated place field firing rates across trials.

    """
    
    # Calculate number of trials to simulate
    if presence_ratio is not None:
        n_simulated_trials = int(n_trials * presence_ratio)
    else:
        n_simulated_trials = n_trials 
    trial_placefield  = []

    for _ in range(n_simulated_trials):
        if vary_height:
            height = np.random.normal(height_mean, height_std)
        else:
            height = height_mean
        
        if vary_width:
            width = np.random.normal(width_mean, width_std)
        else:
            width = width_mean
        
        if vary_place_loc:
            place_loc = np.random.normal(place_loc_mean, place_loc_std)
        else:
            place_loc = place_loc_mean
            
        place_bins = sim_placefield(height, width, n_bins, place_loc, base_mean, base_std, noise_std, plot=False)
   
        trial_placefield.append(place_bins)

    # Fill the remaining trials with empty arrays (zeros)
    for _ in range(n_trials - n_simulated_trials):
        trial_placefield .append(np.zeros(n_bins))

    trial_placefield = abs(np.array(trial_placefield ))
    
    # Calculate presence ratio if not provided (based on all trials, including empty)
    if presence_ratio is None:
        presence_ratio = np.mean(trial_placefield  > 0, axis=0)  # Fraction of non-zero values across trials for each bin

    if plot:
        grid = make_grid(1, 2, wspace=.5, hspace=1, figsize=(15, 5))

        plt.rcParams.update({'font.size': 25})
        ax = get_grid_subplot(grid,0, 0)
        drop_spines(['top','right'],ax=ax)
        plot_trial_placefield(trial_placefield,average='mean',add_traces = False,shade='sem',trace_cmap = 'Greys',ax=ax)
        ax = get_grid_subplot(grid,0,1 )
        plot_trial_placefield(trial_placefield,average='mean',add_traces = True,trace_cmap = 'Greys',ax=ax)
        drop_spines(['top','right'],ax=ax)
        

    return trial_placefield




def sim_skew_trial_placefield(height_mean, height_std, width_mean, width_std, place_loc_mean, place_loc_std, skewness_mean, skewness_std, n_bins, noise_std, base_mean, base_std, n_trials, vary_height=True, vary_width=True, vary_place_loc=True, vary_skewness=True, plot=True, presence_ratio=1):
    """Simulate multiple trials of skewed place fields with optional variability in height, width, location, and skewness.
    
    Parameters
    ----------
    height_mean : float
        Mean peak firing rate for the place field.
    height_std : float
        Standard deviation for the peak firing rate.
    width_mean : float
        Mean width of the place field.
    width_std : float
        Standard deviation for the width of the place field.
    place_loc_mean : float
        Mean location of the center of the place field.
    place_loc_std : float
        Standard deviation for the location of the center of the place field.
    skewness_mean : float
        Mean skewness value of the place field, controlling its asymmetry.
    skewness_std : float
        Standard deviation for the skewness value.
    n_bins : int
        Number of spatial bins over which the place field is simulated.
    noise_std : float
        Standard deviation for the noise added to the place field.
    base_mean : float
        Mean baseline firing rate. 
    base_std : float
        Standard deviation of the baseline firing rate.
    n_trials : int
        Number of trials to simulate.
    vary_height : bool, optional
        If True, the height of the place field varies across trials. Default is True.
    vary_width : bool, optional
        If True, the width of the place field varies across trials. Default is True.
    vary_place_loc : bool, optional
        If True, the location of the place field center varies across trials. Default is True.
    vary_skewness : bool, optional
        If True, the skewness of the place field varies across trials. Default is True.
    plot : bool, optional
        If True, plots the individual trials and their average with error bands. Default is True.
    presence_ratio : float, optional
        The fraction of trials to simulate with data, with the remainder being filled with zeros. Default is 1 (all trials).

    Returns
    -------
    trial_place_bins : 2D array
        Simulated place field firing rates across trials. Each trial is represented by a 1D array.

    """
    if presence_ratio is not None:
        n_simulated_trials = int(n_trials * presence_ratio)  # Calculate number of trials to simulate
    else:
        n_simulated_trials = n_trials  # Default to simulating all trials

    trial_placefield = []

    for _ in range(n_simulated_trials):
        # Vary parameters for each trial if specified
        height = np.random.normal(height_mean, height_std) if vary_height else height_mean
        width = np.random.normal(width_mean, width_std) if vary_width else width_mean
        place_loc = np.random.normal(place_loc_mean, place_loc_std) if vary_place_loc else place_loc_mean
        skewness = np.random.normal(skewness_mean, skewness_std) if vary_skewness else skewness_mean
        
        # Generate skewed place field for this trial
        place_bins = sim_skew_placefield(height, width, skewness, n_bins, place_loc, base_mean, base_std, noise_std, plot=False)
        trial_placefield.append(place_bins)

    # Fill the remaining trials with empty arrays (zeros)
    for _ in range(n_trials - n_simulated_trials):
        trial_placefield.append(np.zeros(n_bins))

    trial_placefield = abs(np.array(trial_placefield))
    
    # Calculate presence ratio if not provided (based on all trials, including empty)
    if presence_ratio is None:
        presence_ratio = np.mean(trial_placefield > 0, axis=0)  # Fraction of non-zero values across trials for each bin

    if plot:
        grid = make_grid(1, 2, wspace=.5, hspace=1, figsize=(15, 5))

        plt.rcParams.update({'font.size': 25})
        ax = get_grid_subplot(grid,0, 0)
        drop_spines(['top','right'],ax=ax)
        plot_trial_placefield(trial_placefield,average='mean',add_traces = False,shade='sem',trace_cmap = 'Greys',ax=ax)
        ax = get_grid_subplot(grid,0,1 )
        plot_trial_placefield(trial_placefield,average='mean',add_traces = True,trace_cmap = 'Greys',ax=ax)
        drop_spines(['top','right'],ax=ax)
        

    return trial_placefield


def sim_trial_multi_placefields(n_height_mean, n_height_std, n_width_mean, n_width_std, n_place_locs_mean, n_place_loc_std, n_bins, n_peaks, base_mean, base_std, noise_std, n_trials, vary_height=True, vary_width=True, vary_place_loc=True, plot=True, presence_ratio=None):
    """ Simulate multiple trials of multi-peak place fields with specified parameters
    
    Parameters
    -----------
    n_height_mean : array-like
        Mean peak firing rates for each place field.
    n_height_std : array-like
        Standard deviation for the peak firing rates for each place field.
    n_width_mean : array-like
        Mean widths for each place field.
    n_width_std : array-like
        Standard deviation for the widths for each place field.
    n_place_locs_mean : array-like
        Mean positions of the centers of each place field.
    n_place_loc_std : array-like
        Standard deviation for the place field centers.
    n_bins : int
        Number of spatial bins.
    n_peaks : int
        Number of peaks to simulate for each trial.
    base_mean : float
        Mean baseline firing rate.
    base_std : float
        Standard deviation of baseline firing rate.
    noise_std : float
        Standard deviation for added noise.
    n_trials : int
        Number of trials to simulate.
    vary_height : bool, optional
        If True, vary the height for each trial.
    vary_width : bool, optional
        If True, vary the width for each trial.
    vary_place_loc : bool, optional
        If True, vary the place field centers for each trial.
    plot : bool, optional
        If True, plot the simulated data.
    presence_ratio : float, optional
        Ratio of trials to simulate, with the rest being empty.

    Returns
    -------
    trial_placefields : 2D array
        Simulated place field firing rates across trials.

    """
    
    if presence_ratio is not None:
        n_simulated_trials = int(n_trials * presence_ratio)  # Calculate number of trials to simulate
    else:
        n_simulated_trials = n_trials  # Default to simulating all trials

    trial_placefields = []

    # Simulate only the specified number of trials
    for _ in range(n_simulated_trials):
        n_height = np.random.normal(n_height_mean, n_height_std) if vary_height else n_height_mean
        n_width = np.random.normal(n_width_mean, n_width_std) if vary_width else n_width_mean
        n_place_loc = np.random.normal(n_place_locs_mean, n_place_loc_std) if vary_place_loc else n_place_locs_mean

        placefield=sim_multi_placefield(n_height, n_width, n_bins, n_place_loc, n_peaks, base_mean, base_std, noise_std,plot=False,)
        trial_placefields.append(placefield)

    # Fill the remaining trials with empty arrays (zeros)
    for _ in range(n_trials - n_simulated_trials):
        trial_placefields.append(np.zeros(n_bins))

    trial_placefields = abs(np.array(trial_placefields))
    
    # Calculate presence ratio if not provided (based on all trials, including empty)
    if presence_ratio is None:
        presence_ratio = np.mean(trial_placefields > 0, axis=0)  # Fraction of non-zero values across trials for each bin

    if plot:
        grid = make_grid(1, 2, wspace=.5, hspace=1, figsize=(15, 5))

        plt.rcParams.update({'font.size': 25})
        ax = get_grid_subplot(grid,0, 0)
        drop_spines(['top','right'],ax=ax)
        plot_trial_placefield(trial_placefields,average='mean',add_traces = False,shade='sem',trace_cmap = 'Greys',ax=ax)
        ax = get_grid_subplot(grid,0,1 )
        plot_trial_placefield(trial_placefields,average='mean',add_traces = True,trace_cmap = 'Greys',ax=ax)
        drop_spines(['top','right'],ax=ax)
        

    return trial_placefields



def sim_trial_multi_skew_placefields(n_height_mean, n_height_std, n_width_mean, n_width_std, n_place_locs_mean, n_place_loc_std, n_skewness_mean, n_skewness_std, n_bins, n_peaks, base_mean, base_std, noise_std, n_trials, vary_height=True, vary_width=True, vary_place_loc=True, vary_skewness=True, plot=True, presence_ratio=None):
    """ Simulate multiple trials of multi-peak place fields with specified parameters, including skewness variability
    
    Parameters
    -----------
    n_height_mean : array-like
        Mean peak firing rates for each place field.
    n_height_std : array-like
        Standard deviation for the peak firing rates for each place field.
    n_width_mean : array-like
        Mean widths for each place field. 
    n_width_std : array-like
        Standard deviation for the widths for each place field. 
    n_place_locs_mean : array-like
        Mean positions of the centers of each place field. 
    n_place_loc_std : array-like
        Standard deviation for the place field centers.  
    n_skewness_mean : array-like
        Mean skewness values for each peak.  
    n_skewness_std : array-like
        Standard deviation for the skewness values for each peak.  
    n_bins : int
        Number of spatial bins.
    n_peaks : int
        Number of peaks to simulate for each trial.
    base_mean : float
        Mean baseline firing rate.
    base_std : float
        Standard deviation of baseline firing rate.
    noise_std : float
        Standard deviation for added noise.
    n_trials : int
        Number of trials to simulate.
    vary_height : bool, optional
        If True, vary the height for each trial. Default is True.
    vary_width : bool, optional
        If True, vary the width for each trial. Default is True.
    vary_place_loc : bool, optional
        If True, vary the place field centers for each trial. Default is True.
    vary_skewness : bool, optional
        If True, vary the skewness for each trial. Default is True.
    plot : bool, optional
        If True, plot the simulated data. Default is True.
    presence_ratio : float, optional
        Ratio of trials to simulate, with the rest being empty. Default is None.

    Returns
    -------
    trial_placefields : 2D array
        Simulated place field firing rates across trials.

    """
    
    if presence_ratio is not None:
        n_simulated_trials = int(n_trials * presence_ratio)  # Calculate number of trials to simulate
    else:
        n_simulated_trials = n_trials  # Default to simulating all trials

    trial_placefields = []

    # Simulate only the specified number of trials
    for _ in range(n_simulated_trials):
        n_height = np.random.normal(n_height_mean, n_height_std) if vary_height else n_height_mean
        n_width = np.random.normal(n_width_mean, n_width_std) if vary_width else n_width_mean
        n_place_loc = np.random.normal(n_place_locs_mean, n_place_loc_std) if vary_place_loc else n_place_locs_mean
        n_skewness = np.random.normal(n_skewness_mean, n_skewness_std) if vary_skewness else n_skewness_mean

        # Generate multi-peak place field with skewness
        placefield = sim_multi_skew_placefield(n_height, n_width, n_bins, n_place_loc, n_peaks, n_skewness, base_mean, base_std, noise_std, plot=False)
        trial_placefields.append(placefield)

    # Fill the remaining trials with empty arrays (zeros)
    for _ in range(n_trials - n_simulated_trials):
        trial_placefields.append(np.zeros(n_bins))

    trial_placefields = abs(np.array(trial_placefields))
    
    # Calculate presence ratio if not provided (based on all trials, including empty)
    if presence_ratio is None:
        presence_ratio = np.mean(trial_placefields > 0, axis=0)  # Fraction of non-zero values across trials for each bin

    if plot:
        grid = make_grid(1, 2, wspace=.5, hspace=1, figsize=(15, 5))

        plt.rcParams.update({'font.size': 25})
        ax = get_grid_subplot(grid,0, 0)
        drop_spines(['top','right'],ax=ax)
        plot_trial_placefield(trial_placefields,average='mean',add_traces = False,shade='sem',trace_cmap = 'Greys',ax=ax)
        ax = get_grid_subplot(grid,0,1 )
        plot_trial_placefield(trial_placefields,average='mean',add_traces = True,trace_cmap = 'Greys',ax=ax)
        drop_spines(['top','right'],ax=ax)
    return trial_placefields
