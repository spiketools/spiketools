"""Tests for spiketools.utils.checks"""

import warnings

from pytest import warns

from spiketools.utils.checks import *

###################################################################################################
###################################################################################################

def test_check_spike_time_unit(tspikes):

    # Make warnings come up as errors, so they can be caught
    warnings.filterwarnings("error")

    # Check test data in ms, that should pass
    check_spike_time_unit(tspikes * 1000)
    warnings.resetwarnings()

    # Check test data in seconds, that should raise an error
    with warns(UserWarning):
        check_spike_time_unit(tspikes)
