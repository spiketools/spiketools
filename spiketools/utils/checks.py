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
        Only used if `bins` is a float. If given, the end value is inclusive.
    check_range : True
        Whether the check the range of the data values against the time bins.

    Returns
    -------
    bins : 1d array
        Time bins.
    """

    if isinstance(bins, (int, float)):
        # Define time range based on data, if not otherwise set
        if not trange:
            trange = [0, np.max(values) + bins]
        # If time range is given, update to include end value
        else:
            trange[1] = trange[1] + bins
        bins = np.arange(*trange, bins)

    elif isinstance(bins, np.ndarray):
        # Check that bins are well defined (monotonically increasing)
        assert np.all(np.diff(bins) > 0), 'Bin definition is ill-formed.'

    # Check that given bin range matches the data values
    if values is not None and values.size > 0:
        if check_range and (np.min(values) < bins[0] or np.max(values) > bins[-1]):
            warnings.warn('The data values extend beyond the given time definition.')

    return bins



def check_position_bins(bins, position=None):
    """Check a bin definition for position binning.

    Parameters
    ----------
    bins : int or list of [int, int]
        The bin definition for dividing up the space. If 1d, can be integer.
        If 2d should be a list, defined as [number of x_bins, number of y_bins].
    position : 1d or 2d array, optional
        Position values across a 1D or 2D space.
        If provided, used to check consistency between position dimensionality and bin defintion.

    Returns
    -------
    bins : list of [int] or list of [int, int]
        Bin definition, after checking, and converted to being a list.

    Raises
    ------
    AssertionError
        Raised if there are any issues with the given bin definition.
    """

    if isinstance(bins, int):
        bins = [bins]

    if isinstance(bins, list):
        for binval in bins:
            assert isinstance(binval, int), 'Bin definition values should be integers.'

    assert len(bins) <= 2, 'Bin definition has too many values (>2).'

    if position is not None:
        msg = 'There is a mismatch between position data and bin definition.'
        assert len(bins) == position.ndim, msg

    return bins
