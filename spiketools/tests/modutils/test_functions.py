"""Tests for spiketools.modutils.functions."""

from spiketools.modutils.functions import *

###################################################################################################
###################################################################################################

def test_get_function_parameters():

    def func(a, b=2, c=None):
        pass

    parameters = get_function_parameters(func)
    assert isinstance(parameters, dict)
    assert list(parameters.keys()) == ['a', 'b', 'c']
    assert parameters['b'].default == 2
