"""Tests for spiketools.utils.checks"""

import warnings

import numpy as np

from spiketools.utils.checks import *

###################################################################################################
###################################################################################################

def test_check_spike_time_unit():
    # two test arrays: one that should throw a warning (spikes_ms), one that should not (spikes_s)
    spikes_ms = np.array([2.81287358, 2.98899132, 3.48063933, 3.69837886, 4.1972945 ,
       5.61169988, 5.99306141, 6.42006449, 8.0689054 , 9.4872388 ])
    spikes_s = np.linspace(1, 400, 10).astype(int)
    # Make warnings come up as errors, so they can be caught
    warnings.filterwarnings("error")
    try:
        check_spike_time_unit(spikes_ms)
    except:
        check_spike_time_unit(spikes_s)
        warnings.resetwarnings()
        pass