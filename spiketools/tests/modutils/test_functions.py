"""Tests for spiketools.modutils.functions."""

from spiketools.modutils.functions import *

###################################################################################################
###################################################################################################

def test_get_function_parameters():

    def func(a, b, c=None):
        pass

    parameters = get_function_parameters(func)
    assert isinstance(parameters, list)
    assert parameters == ['a', 'b', 'c']
