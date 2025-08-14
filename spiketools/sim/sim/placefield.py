import matplotlib.pyplot as plt

from peak import sim_placefield_peak, sim_skew_placefield_peak,sim_placefield_multipeaks,sim_skew_placefield_multipeaks
from noise import sim_baseline,sim_noise

###################################################################################################
###################################################################################################


def sim_placefield(height, width, n_bins, place_loc, base_mean, base_std, noise_std, plot=True):
    """ Simulate place field on a linear track
    
    Parameters
    -----------
    height: int
            Place field's peak firing rate.
    width:  int
            Width of the place field 
    nbins: int
            Number of spatial bins 
    place loc: int
            Center of the place field location 
    base_mean: int
            Average firing rate 
    base_std: int
            Standard deviation of the firing rate
    noise_std: int
            Standard deviation of the firing rate
            
    
    Returns
    -------
    placefield: 1d array
            Simulated symmetrical place field firing rate, in Hz. 
            

    """
        
    
    placefield = sim_placefield_peak(height, width, n_bins, place_loc) + sim_baseline(n_bins, base_mean, base_std)+sim_noise(n_bins, noise_std)
    if plot:
        plt.plot(placefield)
    return placefield 
    
def sim_skew_placefield(height, width, skewness, n_bins, place_loc, base_mean, base_std, noise_std, plot = True):
    """ Simulate place field on a linear track
    
    Parameters
    -----------
    height: int
            Place field's peak firing rate.
    width:  int
            Width of the place field 
    nbins: int
            Number of spatial bins 
    place loc: int
            Center of the place field location 
            
    base_mean: int
            Average firing rate 
            
    base_std: int
            Standard deviation of the firing rate
            
    noise_std: int
            Standard deviation of the firing rate
            
    skewness: int
            Skewness parameter that introduces asymmetry to the place field (Positive skewness values cause the place field to skew to the right, while negative values result in leftward skewing)
            
    
    Returns
    -------
    skew_placefield_peak: 1d array
            Simulated Asymmetrical place field skew peak firing rate, in Hz.  
            

    """
        
    
    skew_placefield = sim_skew_placefield_peak(height, width, n_bins, place_loc, skewness) + sim_baseline(n_bins, base_mean, base_std)+sim_noise(n_bins, noise_std)
    if plot:
        plt.plot(skew_placefield)
    return skew_placefield 

def sim_multi_placefield(n_height, n_width, n_bins, n_place_loc, n_peaks, base_mean, base_std, noise_std, plot=True):
    """ Simulate place field on a linear track
    
    Parameters
    -----------
    n_height : array-like
        An array containing the peak firing rates for each place field.
    n_width : array-like
        An array specifying the widths of each place field.
    num_bins : int
        The number of spatial bins across the linear track.
    n_place_loc : array-like
        An array of integers specifying the centers of each place field.
    num_peaks : int
        The number of place field peaks to simulate.
            
    base_mean: int
            Average firing rate 
            
    base_std: int
            Standard deviation of the firing rate
            
    noise_std: int
            Standard deviation of the firing rate
            
    
    Returns
    -------
    placefield: 1d array
            Simulated multipeak place field firing rate, in Hz. 
            

    """
        
    
    placefield = sim_placefield_multipeaks(n_height, n_width, n_bins, n_place_loc, n_peaks, plot=False) + sim_baseline(n_bins, base_mean, base_std)+sim_noise(n_bins, noise_std)
    if plot:
        plt.plot(placefield)
    return placefield 



def sim_multi_skew_placefield(n_height, n_width, n_bins, n_place_loc, n_peaks, n_skewness, base_mean, base_std, noise_std, plot=True):
    """ Simulate place field on a linear track
    
    Parameters
    -----------
    n_height : array-like
        An array containing the peak firing rates for each place field.
    n_width : array-like
        An array specifying the widths of each place field.
    n_bins : int
        The number of spatial bins across the linear track.
    n_place_loc : array-like
        An array of integers specifying the centers of each place field.
    n_peaks : int
        The number of place field peaks to simulate.
    n_skewness: array-like
        Skewness of the place field 
    base_mean: int
            Average firing rate            
    base_std: int
            Standard deviation of the firing rate      
    noise_std: int
            Standard deviation of the firing rate
            
    
    Returns
    -------
    skewed multi placefield: 1d array
            Simulated skewed multipeak place field firing rate, in Hz. 
            

    """
        
    
    placefield = sim_skew_placefield_multipeaks(n_height, n_width, n_bins, n_place_loc, n_peaks, n_skewness, plot=False) + sim_baseline(n_bins, base_mean, base_std)+sim_noise(n_bins, noise_std)
    if plot:
        plt.plot(placefield)
    return placefield 
