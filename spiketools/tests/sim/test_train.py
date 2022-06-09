"""Tests for spiketools.sim.train"""

from pytest import raises

from spiketools.tests.tsettings import FS, N_SAMPLES

from spiketools.sim.train import *

###################################################################################################
###################################################################################################

def test_sim_spiketrain():

    n_samples = 1000
    for method, param in zip(['prob', 'binom', 'poisson'], [0.1, 0.1, 10]):
        train = sim_spiketrain(param, n_samples, method)

        assert isinstance(train, np.ndarray)
        assert np.all(train < 2)
        assert sum(train) < len(train)

def test_sim_spiketrain_prob():

    # Simulate spike train based on a probability of spiking per sample over time
    p_spiking_1 = np.ones(100)
    sim_spiketrain_1 = sim_spiketrain_prob(p_spiking_1)

    p_spiking_2 = np.array([0.3, 0.5, 0.1, 0.8])
    sim_spiketrain_2 = sim_spiketrain_prob(p_spiking_2)

    # Simulate spike train of size n_samples, based on a probability of spiking per sample
    p_spiking_3 = 0.
    sim_spiketrain_3 = sim_spiketrain_prob(p_spiking_3, N_SAMPLES)

    # Value checks
    assert np.nansum(sim_spiketrain_1) > 75
    assert np.nansum(sim_spiketrain_2) < len(p_spiking_2)
    assert np.nansum(sim_spiketrain_3) == 0
    # Dimension checks
    assert len(sim_spiketrain_1) == len(p_spiking_1)
    assert len(sim_spiketrain_2) == len(p_spiking_2)
    assert len(sim_spiketrain_3) == N_SAMPLES

    with raises(ValueError):
        sim_spiketrain_4 = sim_spiketrain_prob(p_spiking_3)

def test_sim_spiketrain_binom():

    # Simulate spike train based on a probability of spiking per sample over time
    p_spiking_1 = np.ones(100)
    sim_spiketrain_1 = sim_spiketrain_binom(p_spiking_1)

    p_spiking_2 = np.array([0.3, 0.5, 0.1, 0.8])
    sim_spiketrain_2 = sim_spiketrain_binom(p_spiking_2)

    # Simulate spike train of size n_samples, based on a probability of spiking per sample
    p_spiking_3 = 0.
    sim_spiketrain_3 = sim_spiketrain_binom(p_spiking_3, N_SAMPLES)

    # Value checks
    assert np.nansum(sim_spiketrain_1) > 75
    assert np.nansum(sim_spiketrain_2) < len(p_spiking_2)
    assert np.nansum(sim_spiketrain_3) == 0
    # Dimension checks
    assert len(sim_spiketrain_1) == len(p_spiking_1)
    assert len(sim_spiketrain_2) == len(p_spiking_2)
    assert len(sim_spiketrain_3) == N_SAMPLES

    with raises(ValueError):
        sim_spiketrain_4 = sim_spiketrain_binom(p_spiking_3)

def test_sim_spiketrain_poisson():

    # Simulate spike trains with three different rates
    rate_1 = 100
    rate_2 = 3
    rate_3 = 0

    sim_spiketrain_1 = sim_spiketrain_poisson(rate_1, N_SAMPLES, FS)
    sim_spiketrain_2 = sim_spiketrain_poisson(rate_2, N_SAMPLES, FS)
    sim_spiketrain_3 = sim_spiketrain_poisson(rate_3, N_SAMPLES, FS)

    # Value checks
    assert np.nansum(sim_spiketrain_1) > 75
    assert np.nansum(sim_spiketrain_2) < 10
    assert np.nansum(sim_spiketrain_3) == 0
    # Dimension checks
    assert len(sim_spiketrain_1) == N_SAMPLES
    assert len(sim_spiketrain_2) == N_SAMPLES
    assert len(sim_spiketrain_3) == N_SAMPLES
