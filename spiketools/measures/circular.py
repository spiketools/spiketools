"""Functions for dealing with circular data."""

import numpy as np

###################################################################################################
###################################################################################################

def bin_circular(data, bin_width=10):
    """Bin circular data.

    Parameters
    ----------
    data : 1d array
        Data to bin.
    bin_width : float, optional, default: 10
        Width of the bins to use for the histogram.

    Returns
    -------
    bin_edges : 1d array
        Bin edge definitions.
    counts : 1d array
        Count values per bin.

    Notes
    -----
    This function currently only supports data in degrees.
    """

    bin_edges = np.arange(0, 360 + bin_width, bin_width)
    counts, _ = np.histogram(data, bins=bin_edges)

    return bin_edges, counts
