"""Tests for spiketools.stats.permutations"""

from spiketools.stats.permutations import *

###################################################################################################
###################################################################################################

def test_vec_perm():

    data = np.array([[1, 2, 3], [4, 5, 6]])

    out = vec_perm(data)
    assert isinstance(out, np.ndarray)

def test_compute_empirical_pvalue():

    value = 1.0
    surrogates = np.array([0, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2])

    p_value = compute_empirical_pvalue(value, surrogates)
    assert isinstance(p_value, float)

def test_zscore_to_surrogates():

    value = 1.0
    surrogates = np.array([0, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2])

    zscore = zscore_to_surrogates(value, surrogates)
    assert isinstance(zscore, float)
