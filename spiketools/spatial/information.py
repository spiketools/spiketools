"""Measures of spatial information."""

import numpy as np

###################################################################################################
###################################################################################################

def compute_I(spike_x, spike_z, bins, occupancy):
    """Compute spatial information.

    Parameters
    ----------
    spike_x : 1d array
        xx
    spike_z : 1d array
        xx
    bins :
        xx
    occupancy :
        Occupancy.

    Returns
    -------
    I : ??
        xx
    """

    spike_map = np.histogram2d(spike_x, spike_z, bins=bins)[0]
    info = _compute_spatial_information(spike_map, occupancy)

    return info


def skaggs1d(data, occupancy, bins=40):
    """Compute Skaggs SI.

    Parameters
    ----------
    data :
        position
    occupancy : ?
        Occupancy.
    bins : int
        xx

    Returns
    -------
    info : float
        Spike information rate (bits/spike).

    :param xs: spike_position
    :param freq: sampling frequency
    :param min_occ: minimum occupancy
    """

    spike_map = np.histogram(data, bins=bins)[0]
    info = _compute_spatial_information(spike_map, occupancy)

    return info


def _compute_spatial_information(spike_map, occupancy):
    """   """

    # Calculate average firing rate
    fr = np.nansum(spike_map) / np.nansum(occupancy)
    if fr == 0.0:
        return 0.0

    # Compute the occupancy probability, per bin
    occ_prob = occupancy / np.nansum(occupancy)

    # Normalize spiking across space by occupancy
    spike_map_norm = spike_map / occupancy

    # Create mask for selecting nonzero values
    nz = np.nonzero(spike_map_norm)

    # Calculate the spatial information
    info = np.nansum(occ_prob[nz] * spike_map_norm[nz] * np.log2(spike_map_norm[nz] / fr)) / fr

    return info
