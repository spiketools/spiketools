"""Tests for spiketools.utils.options."""

from pytest import raises

from spiketools.utils.options import *

###################################################################################################
###################################################################################################

def test_get_avg_func():

    for avg_type in ['mean', 'median']:
        func = get_avg_func(avg_type)
        assert callable(func)

    with raises(ValueError):
        get_avg_func('not_a_thing')

def test_get_var_func():

    for var_type in ['std', 'var', 'sem']:
        func = get_var_func(var_type)
        assert callable(func)

    with raises(ValueError):
        get_avg_func('not_a_thing')
