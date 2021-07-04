"""Permutation related functions."""

import numpy as np

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
    Code adapted from here:
    https://stackoverflow.com/questions/46859304/
    """

    data_ext = np.concatenate((data, data[:-1]))
    strides = data.strides[0]
    perms = np.lib.stride_tricks.as_strided(data_ext, shape=(n_perms, len(data)),
                                            strides=(strides, strides),
                                            writeable=False).copy()

    return perms
