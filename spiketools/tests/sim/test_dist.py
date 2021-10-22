"""Tests for spiketools.sim.dist"""

from spiketools.sim.dist import *

###################################################################################################
###################################################################################################

def test_sim_spiketrain_binom():
    pass

def test_sim_spiketrain_poisson():

    # Simulate spike trains with three different rates.
    rate_1 = 100
    rate_2 = 3
    rate_3 = 0
    n_samples = 100
    fs = 100
    
    sim_spiketrain_1 = sim_spiketrain_poisson(rate_1, n_samples, fs)
    sim_spiketrain_2 = sim_spiketrain_poisson(rate_2, n_samples, fs)
    sim_spiketrain_3 = sim_spiketrain_poisson(rate_3, n_samples, fs)
    
    # value checks
    assert np.nansum(sim_spiketrain_1) > 75
    assert np.nansum(sim_spiketrain_2) < 10
    assert np.nansum(sim_spiketrain_3) == 0
    # dimension checks
    assert len(sim_spiketrain_1) == n_samples
    assert len(sim_spiketrain_2) == n_samples
    assert len(sim_spiketrain_3) == n_samples