"""Tests for spiketools.utils.checks"""

from spiketools.utils.checks import *

###################################################################################################
###################################################################################################

def test_infer_time_unit(tspikes):

    # Check test data in seconds
    inferred = infer_time_unit(tspikes)
    assert inferred == 'seconds'

    # Check test data in milliseconds
    inferred = infer_time_unit(tspikes * 1000)
    assert inferred == 'milliseconds'
