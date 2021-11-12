"""Tests for spiketools.stats.permutations"""

from spiketools.stats.permutations import *

###################################################################################################
###################################################################################################

def test_vec_perm():

    data = np.array([[1, 2, 3], [4, 5, 6]])

    out = vec_perm(data)
    assert isinstance(out, np.ndarray)

def test_compute_empirical_pvalue(tdata):

    p_value = compute_empirical_pvalue(1.5, tdata)
    assert isinstance(p_value, float)

def test_zscore_to_surrogates(tdata):

    zscore = zscore_to_surrogates(1.5, tdata)
    assert isinstance(zscore, float)
