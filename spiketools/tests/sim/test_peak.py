"""Tests for spiketools.sim.sim.peak"""

from spiketools.sim.peak import (sim_placefield_peak, sim_skew_placefield_peak, 
                                sim_placefield_multipeaks, sim_skew_placefield_multipeaks)
import numpy as np

###################################################################################################
###################################################################################################

def test_sim_placefield_peak():

    height = 10
    width = 10
    n_bins = 100
    place_loc = 50

    placefield_peak = sim_placefield_peak(height, width, n_bins, place_loc)
    assert isinstance(placefield_peak, np.ndarray)
    assert placefield_peak.shape == (n_bins,)   

def test_sim_skew_placefield_peak():

    height = 10
    width = 10
    n_bins = 100
    place_loc = 50
    skewness = 1

    placefield_peak = sim_skew_placefield_peak(height, width, n_bins, place_loc, skewness)
    assert isinstance(placefield_peak, np.ndarray)

def test_sim_placefield_multipeaks():

    n_height = [10, 20, 30]
    n_width = [10, 20, 30]
    n_bins = 100
    n_place_loc = [50, 60, 70]
    n_peaks = 3

    placefield_multipeaks = sim_placefield_multipeaks(n_height, n_width, n_bins, 
    n_place_loc, n_peaks)
    assert isinstance(placefield_multipeaks, np.ndarray)


def test_sim_skew_placefield_multipeaks():

    n_height = [5.0, 5.0, 5.0]
    n_width = [2.0, 2.0, 2.0]
    n_place_loc = [5, 25, 45]
    n_peaks = 3
    n_skewness = [5, 5, 5]
    n_bins = 50

    placefield_multipeaks = sim_skew_placefield_multipeaks(n_height, n_width, 
                n_bins, n_place_loc, n_peaks, n_skewness)
    assert isinstance(placefield_multipeaks, np.ndarray)