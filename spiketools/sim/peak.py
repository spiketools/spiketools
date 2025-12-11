""" Simulate place field's peak. """

import numpy as np
from scipy.stats import skewnorm

###################################################################################################
###################################################################################################

def sim_placefield_peak(height, width, n_bins, place_loc):
    """Simulate place field's peak based on Gaussian distribution on a linear track.

    Parameters
    ----------
    height: int
            Place field's peak firing rate.
    width:  int
            Width of the place field.
    nbins: int
            Number of spatial bins.
    place loc: int
            Center of the place field location.

    Returns
    -------
    placefield_peak: 1d array
            Simulated symmetrical place field peak firing rate, in Hz.
    """

    spatial_bins = np.arange(n_bins)
    placefield_peak = height * np.exp(-0.5 * ((spatial_bins - place_loc) / width) ** 2)
    return placefield_peak


def sim_skew_placefield_peak(height, width, n_bins, place_loc, skewness):
    """Simulate place field's peak based on Skewed Gaussian distribution.
        on a linear track.

    Parameters
    -----------
    height: int
            Place field's peak firing rate.
    width:  int
            Width of the place field.
    nbins: int
            Number of spatial bins.
    place loc: int
            Center of the place field location.
    skewness: int
            Skewness parameter that introduces asymmetry to the place field
            (Positive skewness values cause the place field to skew to the right, while
            negative values result in leftward skewing).

    Returns
    -------
    skew_placefield_peak: 1d array
            Simulated Asymmetrical place field peak firing rate, in Hz.
    """

    spatial_bins = np.arange(n_bins)
    skew_fr= skewnorm.pdf(spatial_bins, a=skewness, loc=place_loc, scale=width)
    skew_placefield_peak = height * (skew_fr / np.max(skew_fr))

    return skew_placefield_peak


def sim_placefield_multipeaks(n_height, n_width, n_bins, n_place_loc, n_peaks):
    """Simulate a place field with multiple peaks based on
    Gaussian distributions on a linear track.

    Parameters
    ----------
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

    Returns
    -------
    placefield_multipeaks : 1D array
        Simulated place field with multiple symmetrical peaks based on Gaussian distributions,
        representing the summed firing rates across the specified locations and widths.
    """

    placefield_multipeaks = 0
    for i in range(n_peaks):
        placefield_peak = sim_placefield_peak(n_height[i], n_width[i], n_bins, n_place_loc[i])
        placefield_multipeaks += placefield_peak
    return placefield_multipeaks


def sim_skew_placefield_multipeaks(n_height, n_width, n_bins, n_place_loc, n_peaks, n_skewness):
    """Simulate a place field with multiple peaks based on Gaussian distributions
       on a linear track.

    Parameters
    ----------
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

    Returns
    -------
    placefield_multipeaks : 1D array
        Simulated place field with multiple symmetrical peaks based on Gaussian distributions,
        representing the summed firing rates across the specified locations and widths.
    """

    skew_placefield_multipeaks = 0
    for i in range(n_peaks):
        skew_placefield_peak = sim_skew_placefield_peak(n_height[i], n_width[i],
                                     n_bins, n_place_loc[i],n_skewness[i])
        skew_placefield_multipeaks += skew_placefield_peak
    return skew_placefield_multipeaks
