"""Measures of spatial information."""

import numpy as np

from spiketools.spatial.occupancy import normalize_bin_counts

###################################################################################################
###################################################################################################

def compute_spatial_information(bin_firing, occupancy, normalize=False):
    """Compute spatial information.

    Parameters
    ----------
    bin_firing : 1d or 2d array
        Binned firing.
    occupancy : 1d or 2d array
        Occupancy across the space.
    normalize : bool, optional, default: False
        If True, normalize the binned firing rate data by the occupancy.
        If False, it is assumed that the binned firing has already been normalized.

    Returns
    -------
    info : float
        Spike information rate for spatial information (bits/spike).

    Notes
    -----
    This measure computes the spatial information between the firing and spatial location, as:

    .. math::

        I = \\sum{\\lambda (x) log_2 \\frac{\\lambda(x)} {\\lambda} p(x)dx}

    References
    ----------
    .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1992). An
           Information-Theoretic Approach to Deciphering the Hippocampal Code.
           Advances in neural information processing systems.

    Examples
    --------
    Compute spatial information across a 1d space:

    >>> bin_firing = np.array([1, 1, 1, 1, 4])
    >>> occupancy = np.array([1, 1, 1, 1, 1])
    >>> info = compute_spatial_information(bin_firing, occupancy)
    >>> print('{:5.4f}'.format(info))
    0.3219

    Compute spatial information across a 2d space:

    >>> bin_firing = np.array([[1, 1, 1, 5],
    ...                        [1, 1, 1, 5]])
    >>> occupancy = np.array([[1, 1, 1, 1],
    ...                       [1, 1, 1, 1]])
    >>> info = compute_spatial_information(bin_firing, occupancy)
    >>> print('{:5.4f}'.format(info))
    0.4512
    """

    if normalize:
        bin_firing = normalize_bin_counts(bin_firing, occupancy)

    # Calculate average firing rate of the neuron, dividing out by total occupancy time
    #   Note: this recomputes total spike (basically, de-normalizing)
    rate = np.nansum(bin_firing * occupancy) / np.nansum(occupancy)

    # Catch for a neuron with no firing - return 0 information
    if rate == 0.0:
        return 0.0

    # Compute the occupancy probability, per bin
    occ_prob = occupancy / np.nansum(occupancy)

    # Calculate the spatial information, using a mask for nonzero values
    nz = np.nonzero(bin_firing)
    info = np.nansum(occ_prob[nz] * bin_firing[nz] * np.log2(bin_firing[nz] / rate)) / rate

    return info
