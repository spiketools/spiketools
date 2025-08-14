###
import numpy as np 
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def _sim_random(mean, std, n_bins):    
    return np.random.normal(mean, std, size=n_bins)

def sim_baseline(n_bins,base_mean,base_std,plot = False):
    
    """ Simulate place field's baseline firing rate - randomization
    
    Parameters
    -----------
    num_bins: int
            Number of spatial bins 
            
    base_mean: int
            Average firing rate 
            
    base_std: int
            Standard deviation of the firing rate

    Returns
    -------
    baseline: 1d array
            Simulated baseline firing rate, in Hz.
    """
    
    
    baseline = _sim_random(base_mean, base_std, n_bins)
    if plot:
        plt.plot(baseline)
    return baseline

def sim_noise(n_bins, noise_std, plot = False):
    """ Simulate place field's baseline firing rate - randomization
    
    Parameters
    -----------
    num_bins: int
            Number of spatial bins 
              
    noise_std: int
            Standard deviation of the firing rate

    Returns
    -------
    noise: 1d array
            Simulated noise, in Hz.
    """

    noise =  _sim_random(0, noise_std, n_bins)
    if plot:
        plt.plot(noise)
    return noise
    