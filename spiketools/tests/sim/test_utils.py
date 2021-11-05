"""Tests for spiketools.sim.utils"""

from spiketools.sim.utils import *

###################################################################################################
###################################################################################################

def test_refractory():
    
    spikes = np.array([0, 1, 1, 0, 1])
    refractory_time = 0.002
    fs = 1000

    refractory_spikes = refractory(spikes, refractory_time, fs)

    # output dimension check
    assert refractory_spikes.shape == spikes.shape

    # output value check - check all elements are 0 or 1
    assert np.isin(spikes, [0, 1]).all()

    # output accuracy check - check spike within refractory period is dropped
    assert np.array_equal(refractory_spikes, np.array([0, 1, 0, 0, 1]))
