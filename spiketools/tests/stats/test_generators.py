"""Tests for spiketools.stats.generators"""

import inspect

from spiketools.stats.generators import *

###################################################################################################
###################################################################################################

def test_poisson_generator():

    ptrain = poisson_generator(5., 10.)

    assert inspect.isgenerator(ptrain)

    val = next(ptrain)
    assert isinstance(val, float)
