"""Tests for spiketools.stats.generators"""

import inspect

from spiketools.stats.generators import *

###################################################################################################
###################################################################################################

def test_poisson_train():

    ptrain = poisson_train(5., 10.)

    assert inspect.isgenerator(ptrain)

    val = next(ptrain)
    assert isinstance(val, float)
