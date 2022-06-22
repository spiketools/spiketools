"""Permutation related functions."""

import numpy as np
from scipy.stats import zmap

from spiketools.plts.stats import plot_surrogates

###################################################################################################
###################################################################################################

def vec_perm(data, n_perms=1000):
    """Vectorized permutations.

    Parameters
    ----------
    data : 1d array
        Data to permute.
    n_perms : int, optional, default: 1000
        Number of permutations to do.

    Returns
    -------
    perms : 2d array
        Permutations of the input data.

    Notes
    -----
    Code adapted from here: https://stackoverflow.com/questions/46859304/
    This function doesn't have any randomness: for a given array it will
    iterate through the same set of permutations.
    This does a sequence of rotated permutations.

    Examples
    --------
    Create 4 permutations for vector: [0, 5, 10, 15, 20]

    >>> vec = np.array([0, 5, 10, 15, 20])
    >>> vec_perm(vec, n_perms=4)
    array([[ 0,  5, 10, 15, 20],
           [ 5, 10, 15, 20,  0],
           [10, 15, 20,  0,  5],
           [15, 20,  0,  5, 10]])
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

    Examples
    --------
    Compute empirical p-value of value given surrogates is 100 samples from normal distribution.

    >>> value = 0.9
    >>> surrogates = np.random.normal(size=100)
    >>> pval = compute_empirical_pvalue(value, surrogates)
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

    Examples
    --------
    Compute z-score of value given surrogates is 100 samples from normal distribution.

    >>> value = 0.9
    >>> surrogates = np.random.normal(size=100)
    >>> zscore = zscore_to_surrogates(value, surrogates)
    """

    return zmap(value, surrogates)[0]


def compute_surrogate_stats(data_value, surrogates, plot=False, verbose=False, **plt_kwargs):
    """Compute surrogate statistics.

    Parameters
    ----------
    data_value : float
        Test value.
    surrogates : 1d array
        Distribution of surrogates values.
    plot : bool, optional, default: False
        Whether to display the plot of the surrogates values.
    verbose : bool, optional, default: False
        Whether to print the values of the p-value and z-score.

    Returns
    -------
    p_val : float
        The empirical p-value of the test value, as compared to the surrogates.
    z_score : float
        The z-score of the test value, as compared to the surrogates.
    """

    p_val = compute_empirical_pvalue(data_value, surrogates)
    z_score = zscore_to_surrogates(data_value, surrogates)

    if plot:
        plot_surrogates(surrogates, data_value, p_val, **plt_kwargs)

    if verbose:
        print('p-value: {:4.2f}'.format(p_val))
        print('z-score: {:4.2f}'.format(z_score))

    return p_val, z_score
