"""Tests for spiketools.utils.select."""

from pytest import raises

from spiketools.utils.select import *

###################################################################################################
###################################################################################################

def test_get_avg_func():

    func = get_avg_func('mean')
    assert callable(func)

    func = get_avg_func('median')
    assert callable(func)

    with raises(ValueError):
        get_avg_func('not_a_thing')
