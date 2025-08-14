"""Utilities for managing and updating simulation parameters"""


import numpy as np 
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

## Define update functions 

# Update Heights
"""Update the height of the place field.
"""
upd_height = lambda params, val : params.update({'height_mean' : val})

# Update Width 
"""Update the width of the place field.
"""
upd_width = lambda params, val: params.update({'width_mean': val})

# Update Noise
"""Update the noise of the place field.
"""
upd_noise = lambda params, val: params.update({'noise_std':val})

# Update Place field Center Consistency location 
"""Update the place field center consistency location.
"""
upd_placeloc = lambda params, val: params.update({'place_loc_std': val})

# Update Skewness
"""Update the skewness of the place field.
"""
upd_skewness = lambda params, val: params.update({'skewness_mean': val})

# Update Presence Ratio
"""Update the presence ratio of the place field.
"""
upd_presence_ratio = lambda params, val: params.update({'presence_ratio': val})

# Update Base
"""Update the baseline firing rate of the place field.
"""
upd_base = lambda params, val: params.update({'base_mean':val})

# Update Trials
"""Update the number of trials.
"""
upd_trials = lambda params, val: params.update({'n_trials':val})

# Update Number of Peaks
def upd_npeaks(params, val):
    """Update number of peaks and corresponding parameter arrays for multiple place fields.
    
    Parameters
    ----------
    params: dict
        Dictionary of parameters
    val: int
        Number of peaks

    Returns
    -------
    params: dict
        Dictionary of parameters.
    """
    # Calculate evenly spaced locations across the spatial bins
    n_bins = params['n_bins']
    spacing = n_bins / (val + 1)  # Add 1 to val to create margins at edges
    locations = [int(spacing * (i + 1)) for i in range(val)]
    
    params.update({
        'n_peaks': val,
        'n_height_mean': [params['n_height_mean'][0]] * val,
        'n_height_std': [params['n_height_std'][0]] * val,
        'n_width_mean': [params['n_width_mean'][0]] * val,
        'n_width_std': [params['n_width_std'][0]] * val,
        'n_place_locs_mean': locations,  # Evenly spaced locations based on n_bins
        'n_place_loc_std': [params['n_place_loc_std'][0]] * val
    })
    return params

def upd_skew_npeaks(params, val):
    """Update number of peaks and corresponding parameter arrays for multiple place fields.

    Parameters
    ----------
    params: dict
        Dictionary of parameters
    val: int
        Number of peaks

    Returns
    -------
    params: dict
        Dictionary of parameters.
    """
    # Calculate evenly spaced locations across the spatial bins
    n_bins = params['n_bins']
    spacing = n_bins / (val + 1)  # Add 1 to val to create margins at edges
    locations = [int(spacing * (i + 1)) for i in range(val)]
    
    params.update({
        'n_peaks': val,
        'n_height_mean': [params['n_height_mean'][0]] * val,
        'n_height_std': [params['n_height_std'][0]] * val,
        'n_width_mean': [params['n_width_mean'][0]] * val,
        'n_width_std': [params['n_width_std'][0]] * val,
        'n_skewness_mean': [params['n_skewness_mean'][0]] * val,
        'n_skewness_std': [params['n_skewness_std'][0]] * val,
        'n_place_locs_mean': locations,  # Evenly spaced locations based on n_bins
        'n_place_loc_std': [params['n_place_loc_std'][0]] * val
    })
    return params

###########################################################
## Collect together all update function 

UPDATES = {
'update_height' : upd_height,
'update_width': upd_width,
'update_noise': upd_noise,     
'update_placeloc':upd_placeloc,
'update_skewness':upd_skewness,
'update_base':upd_base,
'update_npeaks':upd_npeaks,
'update_skew_npeaks':upd_skew_npeaks,
'update_presence_ratio':upd_presence_ratio
}

## Functions for updating / sampling parameters 
def update_vals(sim_params, values, update):
    """Update simulation parameter values.
    
    Parameters
    ----------
    sim_params: dict
        Dictionary of parameters
    values: list
        List of values to update
    update: function
        Function to update the parameters

    Returns
    -------
    sim_params: dict
        Dictionary of parameters.
    """

    for val in values:
        update(sim_params, val)
        yield sim_params


        
## Functions for updating / sampling parameters 
def update_paired_vals(sim_params, values1, values2, update1,  update2):
    """Update simulation parameter values for paired parameters.
    
    Parameters
    ----------
    sim_params: dict
        Dictionary of parameters
    values1: list
        List of values to update
    values2: list
        List of values to update
    update1: function
        Function to update the first parameter
    update2: function
        Function to update the second parameter

    Returns
    -------
    sim_params: dict
        Dictionary of parameters
    """

    for v1 in values1:
        update1(sim_params, v1)
        for v2 in values2:
            update2(sim_params, v2)
            yield sim_params
            
def sampler(sample_size,min_val,max_val,plot = True):
    """Sample values from a uniform distribution.
    
    Parameters
    ----------
    sample_size: int
        Number of samples to draw
    min_val: float
        Minimum value
    max_val: float
        Maximum value
    plot: bool
        Whether to plot the samples

    Returns
    -------
    sample_vals: array-like
        Sampled values.
    """

    sample_vals = np.random.uniform(min_val, max_val, sample_size)
    if plot:
        norm = plt.Normalize(sample_vals.min(), sample_vals.max())
        cmap = plt.get_cmap('inferno')


        plt.scatter(range(1, sample_size + 1), sample_vals, c=sample_vals, cmap=cmap, marker='o',s=20,norm = norm)
        plt.xlabel('Sample Size')
        plt.ylabel('Sampled Values')
        plt.colorbar(label='Sample Values')
        plt.show()
    return sample_vals
