"""Tests for spiketools.measures.trials"""

import numpy as np

from spiketools.measures.trials import *

###################################################################################################
###################################################################################################

def test_compute_trial_frs(tspikes):

    trial_spikes = [tspikes, tspikes]
    bins = np.arange(0, 10 + 0.5, 0.5)
    out = compute_trial_frs(trial_spikes, bins)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(trial_spikes), len(bins) - 1)
