"""Permutation related functions."""

import numpy as np
from scipy.stats import zmap

###################################################################################################
###################################################################################################

def vec_perm(data, n_perms=1000):
    """Vectorized permutations.

    Parameters
    ----------
    data :
        xx
    n_perms : int
        Number of permutations to do.

    Returns
    -------
    perms :
        xx

    Notes
    -----
    Code adapted from here: https://stackoverflow.com/questions/46859304/
    This function doesn't have any randomness: for a given array it will
    iterate through the same set of permutations.
    This does a sequence of rotated permutations.
    """

    data_ext = np.concatenate((data, data[:-1]))
    strides = data.strides[0]
    perms = np.lib.stride_tricks.as_strided(data_ext, shape=(n_perms, len(data)),
                                            strides=(strides, strides),
                                            writeable=False).copy()

    return perms


def compute_empirical_pvalue(value, surrogates):
    """Compute the empirical p-value from a distribution of surrogates.

    Parameters
    ----------
    value : float
        Test value.
    surrogates : 1d array
        Distribution of surrogates.

    Returns
    -------
    float
        The empirical p-value.
    """

    return sum(surrogates > value) / len(surrogates)


def zscore_to_surrogates(value, surrogates):
    """Z-score a computed value relative to a distribution of surrogates.

    Parameters
    ----------
    value : float
        Value to z-score.
    surrogates : 1d array
        Distribution of surrogates to compute the z-score against.

    Returns
    -------
    float
        The z-score of the given value.
    """

    return zmap(value, surrogates)[0]
