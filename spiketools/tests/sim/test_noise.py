"""Tests for spiketools.sim.noise"""

from spiketools.sim.noise import sim_baseline, sim_noise
import numpy as np

###################################################################################################
###################################################################################################

def test_sim_baseline():

    n_bins = 100
    base_mean = 5.0
    base_std = 0.0

    baseline = sim_baseline(n_bins, base_mean, base_std)
    assert isinstance(baseline, np.ndarray)
    assert baseline.shape == (n_bins,)
    assert np.mean(baseline) == base_mean
 

def test_sim_noise():

    n_bins = 100
    noise_std = 2

    noise = sim_noise(n_bins, noise_std)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (n_bins,)

