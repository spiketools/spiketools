"""Spike related checking functions."""

import warnings

import numpy as np

###################################################################################################
###################################################################################################

def infer_time_unit(time_values):
    """Infer the time unit of given time values.

    Parameters
    ----------
    time_values : 1d array
        Time values.

    Returns
    -------
    time_unit : {'seconds', 'milliseconds'}
        The inferred time unit of the input data.
    """

    time_unit = None

    # Infer seconds if there are any two spikes within the same time unit,
    if len(np.unique((time_values).astype(int))) < len(np.unique(time_values)):
        time_unit = 'seconds'

    # Infer seconds if the mean time between spikes is low
    elif np.mean(np.diff(time_values)) < 10:
        time_unit = 'seconds'

    # Otherwise, infer milliseconds
    else:
        time_unit = 'milliseconds'

    return time_unit


def check_time_bins(bins, values, trange=None, check_range=True):
    """Check a given time bin definition, and define if only given a time resolution.

    Parameters
    ----------
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    values : 1d array
        The time values that are to be binned.
    trange : list of [float, float]
        Time range, in seconds, to create the binned firing rate across.
        Only used if `bins` is a float.
    check_range : True
        Whether the check the range of the data values against the time bins.

    Returns
    -------
    bins : 1d array
        Time bins.
    """

    if isinstance(bins, float):
        trange = [0, np.max(values) + bins] if not trange else trange
        bins = np.arange(*trange, bins)

    elif isinstance(bins, np.ndarray):
        # Check that bins are well defined (monotonically increasing)
        assert np.all(np.diff(bins) > 0), 'Bin definition is ill-formed.'

    # Check that given bin range matches the data values
    if check_range and (np.min(values) < bins[0] or np.max(values) > bins[-1]):
        warnings.warn('The data values extend beyond the given time definition.')

    return bins
