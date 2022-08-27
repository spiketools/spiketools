"""General purpose checker functions."""

import warnings

import numpy as np

###################################################################################################
###################################################################################################

def check_param_range(param, label, bounds):
    """Check a parameter value is within an acceptable range.

    Parameters
    ----------
    param : float
        Parameter value to check.
    label : str
        Label of the parameter being checked.
    bounds : list of [float, float]
       Bounding range of valid values for the given parameter.

    Raises
    ------
    ValueError
        If a parameter that is being checked is out of range.
    """

    if (param < bounds[0]) or (param > bounds[1]):
        msg = "The provided value for the {} parameter is out of bounds. ".format(label) + \
        "It should be between {:1.1f} and {:1.1f}.".format(*bounds)
        raise ValueError(msg)


def check_param_options(param, label, options):
    """Check a parameter value is one of the acceptable options.

    Parameters
    ----------
    param : str
        Parameter value to check.
    label : str
        Label of the parameter being checked.
    options : list of str
        Valid string values that `param` may be.

    Raises
    ------
    ValueError
        If a parameter that is being checked is not in `options`.
    """

    if param not in options:
        msg = "The provided value for the {} parameter is invalid. ".format(label) + \
        "It should be chosen from {{{}}}.".format(str(options)[1:-1])
        raise ValueError(msg)


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

    Examples
    --------
    Infer the time unit of an array of 5 time values:

    >>> time_values = np.array([0.002, 0.01, 0.05, 0.1, 2])
    >>> infer_time_unit(time_values)
    'seconds'
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


def check_bin_range(values, bin_area):
    """Checks data values against given bin edges, warning if values exceed bin range.

    Parameters
    ----------
    values : 1d array
        A set of value to check against bin edges.
    bin_area : 1d array or list
        The bin range area to check. Can be a two-item area range, or an array of bin edges.
    """

    if values.size > 0:
        if np.nanmin(values) < bin_area[0] or np.nanmax(values) > bin_area[-1]:
            msg = 'The data values extend beyond the given bin definition.'
            warnings.warn(msg)


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

    Examples
    --------
    Check if the range of data values exceeds the the 0.5 time bin:

    >>> bins = 0.5
    >>> values = np.array([0.2, 0.4, 0.6, 0.9, 1.4, 1.5, 1.6, 2.0])
    >>> trange = [0.1, 2.1]
    >>> check_time_bins(bins, values, trange, check_range=True)
    array([0.1, 0.6, 1.1, 1.6, 2.1])
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
        if check_range:
            check_bin_range(values, bins)

    return bins
