"""Tests for spiketools.sim.prob"""

from spiketools.sim.prob import *

###################################################################################################
###################################################################################################

def test_sim_spiketrain_prob():

    # Simulate spike train based on a probability of spiking per sample over time.
    p_spiking_1 = np.zeros(10)
    sim_spiketrain_1 = sim_spiketrain_prob(p_spiking_1)
    
    p_spiking_2 = np.array([0.3, 0.5, 0.1, 0.8])
    sim_spiketrain_2 = sim_spiketrain_prob(p_spiking_2)
    
    # Simulate spike train of size n_samples, based on a probability of spiking per sample.
    p_spiking_3 = 1.
    n_samples = 10
    sim_spiketrain_3 = sim_spiketrain_prob(p_spiking_3, n_samples)
    
    # value checks
    assert np.nansum(sim_spiketrain_1) == 0
    assert np.nansum(sim_spiketrain_3) == n_samples
    # dimension checks
    assert len(sim_spiketrain_1) == len(p_spiking_1)
    assert len(sim_spiketrain_2) == len(p_spiking_2)
    assert len(sim_spiketrain_3) == n_samples
    
    try:
        sim_spiketrain_4 = sim_spiketrain_prob(p_spiking_3)
    except ValueError:
        pass