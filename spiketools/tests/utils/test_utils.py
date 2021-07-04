"""Tests for spiketools.utils.utils"""

from spiketools.utils.utils import *

###################################################################################################
###################################################################################################

def test_set_random_seed():

    set_random_seed()
    set_random_seed(100)
