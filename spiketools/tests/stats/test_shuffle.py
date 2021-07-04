"""Tests for spiketools.stats.shuffle"""

from spiketools.stats.shuffle import *

###################################################################################################
###################################################################################################

def test_shuffle_isis(tspikes):

    shuffled = shuffle_isis(tspikes)
    assert True

def test_shuffle_bins(tspikes):

    shuffled = shuffle_bins(tspikes)
    assert True

def test_shuffle_poisson(tspikes):

    shuffled = shuffle_poisson(tspikes)
    assert True

def test_shuffle_circular(tspikes):

    shuffled = shuffle_circular(tspikes, 200)
    assert True
