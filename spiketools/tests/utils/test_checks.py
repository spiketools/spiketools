"""Tests for spiketools.utils.checks"""

from pytest import raises, warns

from spiketools.utils.checks import *

###################################################################################################
###################################################################################################

def test_infer_time_unit(tspikes):

    # Check test data in seconds
    inferred = infer_time_unit(tspikes)
    assert inferred == 'seconds'

    # Check test data in milliseconds
    inferred = infer_time_unit(tspikes * 1000)
    assert inferred == 'milliseconds'

def test_check_time_bins(tspikes):

    # Check precomputed time bins
    tbins = np.arange(0, 10, 0.5)
    out = check_time_bins(tbins, tspikes)
    assert np.array_equal(tbins, out)

    # Check time bins given a time resolution, which should create same bins as precomputed
    out = check_time_bins(0.5, tspikes, trange=[0, 10])
    assert np.array_equal(tbins, out)

    # Check error & warning
    with raises(AssertionError):
        out = check_time_bins(np.array([1, 2, 1]), tspikes, trange=[0, 5])
    with warns(UserWarning):
        out = check_time_bins(0.5, tspikes, trange=[0, 5])
