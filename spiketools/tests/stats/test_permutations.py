"""Tests for spiketools.stats.permutations"""

from spiketools.stats.permutations import *

###################################################################################################
###################################################################################################

def test_compute_empirical_pvalue(tdata):

    p_value = compute_surrogate_pvalue(1.5, tdata)
    assert isinstance(p_value, float)

def test_compute_surrogate_zscore(tdata):

    z_score = compute_surrogate_zscore(1.5, tdata)
    assert isinstance(z_score, float)

def test_compute_surrogate_stats(tdata):

    p_value, z_score = compute_surrogate_stats(1.5, tdata, True, True)
    assert isinstance(p_value, float)
    assert isinstance(z_score, float)
