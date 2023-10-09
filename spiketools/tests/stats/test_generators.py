"""Tests for spiketools.stats.generators"""

import inspect

from spiketools.stats.generators import *

###################################################################################################
###################################################################################################

def test_poisson_generator():

    rate = 10
    duration = 2

    # Test as a generator
    ptrain = poisson_generator(rate, duration)
    assert inspect.isgenerator(ptrain)
    val = next(ptrain)
    for val in ptrain:
        assert isinstance(val, float)

    # Test collection of values matches duration range
    outputs = np.array(list(poisson_generator(rate, duration)))
    assert np.all(outputs > 0)
    assert np.all(outputs < duration)

    # Test collection of values matches duration range, given a start time (positive)
    start_time = 2
    outputs = np.array(list(poisson_generator(rate, duration, start_time)))
    assert np.all(outputs > start_time)
    assert np.all(outputs < start_time + duration)

    # Test collection of values matches duration range, given a start time (negative)
    start_time = -2
    outputs = np.array(list(poisson_generator(rate, duration, start_time)))
    assert np.all(outputs > start_time)
    assert np.all(outputs < start_time + duration)
