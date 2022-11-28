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

def test_get_function_argument():

    def func(a, b=2, c=None):
        pass

    # test accessing from keyword argument
    argument = get_function_argument('b', func, args=[], kwargs={'b' : 4})
    assert argument == 4

    # test accessing from arg
    argument = get_function_argument('b', func, args=[2, 0, 3], kwargs={}, argind=1)
    assert argument == 0

    # test accessing from default
    argument = get_function_argument('b', func, args=[], kwargs={})
    assert argument == 2
