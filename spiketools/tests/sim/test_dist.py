"""Tests for spiketools.sim.dist"""

from pytest import raises

from spiketools.sim.dist import *

###################################################################################################
###################################################################################################

def test_sim_spiketrain_binom():

    # Simulate spike train based on a probability of spiking per sample over time
    p_spiking_1 = np.ones(100)
    sim_spiketrain_1 = sim_spiketrain_binom(p_spiking_1)

    p_spiking_2 = np.array([0.3, 0.5, 0.1, 0.8])
    sim_spiketrain_2 = sim_spiketrain_binom(p_spiking_2)

    # Simulate spike train of size n_samples, based on a probability of spiking per sample
    p_spiking_3 = 0.
    n_samples = 100
    sim_spiketrain_3 = sim_spiketrain_binom(p_spiking_3, n_samples)

    # Value checks
    assert np.nansum(sim_spiketrain_1) > 75
    assert np.nansum(sim_spiketrain_2) < len(p_spiking_2)
    assert np.nansum(sim_spiketrain_3) == 0
    # Dimension checks
    assert len(sim_spiketrain_1) == len(p_spiking_1)
    assert len(sim_spiketrain_2) == len(p_spiking_2)
    assert len(sim_spiketrain_3) == n_samples

    with raises(ValueError):
        sim_spiketrain_4 = sim_spiketrain_binom(p_spiking_3)

def test_sim_spiketrain_poisson():

    # Simulate spike trains with three different rates
    rate_1 = 100
    rate_2 = 3
    rate_3 = 0
    n_samples = 100
    fs = 100

    sim_spiketrain_1 = sim_spiketrain_poisson(rate_1, n_samples, fs)
    sim_spiketrain_2 = sim_spiketrain_poisson(rate_2, n_samples, fs)
    sim_spiketrain_3 = sim_spiketrain_poisson(rate_3, n_samples, fs)

    # Value checks
    assert np.nansum(sim_spiketrain_1) > 75
    assert np.nansum(sim_spiketrain_2) < 10
    assert np.nansum(sim_spiketrain_3) == 0
    # Dimension checks
    assert len(sim_spiketrain_1) == n_samples
    assert len(sim_spiketrain_2) == n_samples
    assert len(sim_spiketrain_3) == n_samples
