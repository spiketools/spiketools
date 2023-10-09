"""Tests for spiketools.utils.random"""

import numpy as np

from spiketools.utils.random import *

###################################################################################################
###################################################################################################

def test_set_random_seed():

    set_random_seed()

    seed = 123
    set_random_seed(seed)
    assert seed == np.random.get_state()[1][0]

def test_set_random_state():

    rng = set_random_state()
    isinstance(rng, np.random.RandomState)

    seed = 234
    rng = set_random_state(seed)
    assert seed == rng.get_state()[1][0]
