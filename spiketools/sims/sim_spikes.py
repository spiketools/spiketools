"""Simulate spikes based on different probability distributions."""

import numpy as np

###################################################################################################
###################################################################################################

def sim_spiketrain_binom1(n_samples, p_spiking):
    """

    Parameters
    ----------

    Returns
    -------
    1d array
        xx
    """

    return np.random.binomial(1, p=p_spiking, size=n_samples)


def sim_spiketrain_binom2(probs):
    """Simulate binary spike train from binomial probability distribution

    Parameters
    ----------
    probs : xx
        xx

    Results
    -------
    xx : xx
        xx
    """

    binary_spktrn = []

    for i in range(len(probs)):
        binary_spktrn.append(np.random.binomial(1, probs[i]))

    return binary_spktrn
