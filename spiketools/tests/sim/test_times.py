"""Tests for spiketools.sim.times"""

from spiketools.sim.times import *

###################################################################################################
###################################################################################################

def test_sim_spiketimes():

    duration = 2
    for method, param in zip(['poisson'], [10]):
        times = sim_spiketimes(param, duration, method)

        assert isinstance(times, np.ndarray)
        assert np.all(times < duration)

def test_sim_spiketimes_poisson():

    rate = 10
    duration = 2
    times = sim_spiketimes_poisson(rate, duration)

    assert isinstance(times, np.ndarray)
    assert np.all(times < duration)
