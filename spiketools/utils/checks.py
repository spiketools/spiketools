"""General purpose checker functions."""

import warnings
from copy import deepcopy

import numpy as np

from spiketools.utils.base import lower_list

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


def check_param_options(param, label, options, ignore_case=False):
    """Check a parameter value is one of the acceptable options.

    Parameters
    ----------
    param : str
        Parameter value to check.
    label : str
        Label of the parameter being checked.
    options : list of str
        Valid string values that `param` may be.
    ignore_case : bool, optional, default: False
        If True, ignore whether strings are upper or lower case for comparison.

    Raises
    ------
    ValueError
        If a parameter that is being checked is not in `options`.
    """

    if ignore_case:
        options = lower_list(options)
        param = param.lower()

    if param not in options:
        msg = "The provided value for the {} parameter is invalid. ".format(label) + \
        "It should be chosen from {{{}}}.".format(str(options)[1:-1])
        raise ValueError(msg)


def check_param_lengths(params, labels, expected_length=None):
    """Check that a set of parameters have the same length.

    Parameters
    ----------
    params : list of collections
        A set of parameters to check that they each have the same length.
    labels : list of str
        The names of the parameters, to print in the error message.
    expected_length : int, optional
        The expected length of each of the parameters, to check against.

    Raises
    ------
    ValueError
        If the parameters are not the same length and/or are not the expected length.
    """

    plen = len(params[0])
    for param in params[1:]:
        if len(param) != plen:
            msg = "These parameters should be the same length: {}.".format(str(labels)[1:-1])
            raise ValueError(msg)

    if expected_length:
        if plen != expected_length:
            msg = "These parameters should all have length {}: {}.".format(\
                expected_length, str(labels)[1:-1])
            raise ValueError(msg)


def check_list_options(contents, label, options):
    """Check a list of values that each element is one of a set of acceptable options.

    Parameters
    ----------
    contents : list of str
        List of values to check
    label : str
        Label of the parameter being checked.
    options : list of str
        Valid string values that each element of `contents` may be.

    Raises
    ------
    ValueError
        If an element of `contents` is not in `options`.
    """

    for el in contents:
        check_param_options(el, label, options)


def check_array_orientation(arr):
    """Check the orientation of an array of data.

    Parameters
    ----------
    arr : ndarray
        Data array to check the orientation of.

    Returns
    -------
    orientation : {'vector', 'row', 'column'}
        The inferred orientation of the data array.
        For 1d cases, 'vector' indicates data is vector with no directional orientation.
        For 2d or 3d cases, 'row' or 'column' indicates data is organized row-wise or column-wise respectively.
    """

    assert arr.ndim < 4, "The check_array_orientation function only works up to 3d."

    if arr.ndim == 1:
        orientation = 'vector'
    # This covers 2d or 3d arrays
    else:
        shape = arr.shape
        if shape[-1] > shape[-2]:
            orientation = 'row'
        else:
            orientation = 'column'

    return orientation


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


def check_time_bins(bins, values=None, time_range=None, check_range=True):
    """Check a given time bin definition, and define if only given a time resolution.

    Parameters
    ----------
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    values : 1d array, optional
        The time values that are to be binned.
        Optional if time range is provided instead.
    time_range : list of [float, float], optional
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
    Check a time bin definition, where bins are defined as a bin length:

    >>> bins = 0.5
    >>> values = np.array([0.2, 0.4, 0.6, 0.9, 1.4, 1.5, 1.6, 1.9])
    >>> time_range = [0., 2.]
    >>> check_time_bins(bins, values, time_range)
    array([0. , 0.5, 1. , 1.5, 2. ])

    Check a time bin definition, where bins are already defined:

    >>> bins = np.array([0. , 0.5, 1. , 1.5, 2. ])
    >>> check_time_bins(bins, values, time_range)
    array([0. , 0.5, 1. , 1.5, 2. ])
    """

    # Take a copy of `time_range` (otherwise, can get an aliasing problem)
    time_range = deepcopy(time_range)

    if isinstance(bins, (int, float)):
        # If time range is given, update to include end value
        if time_range:
            time_range[1] = time_range[1] + bins
        # Otherwise, define time range based on data
        else:
            assert values is not None, "check_time_bins: either `values` or `time_range` required"
            time_range = [0, np.max(values) + bins]
        bins = np.arange(*time_range, bins)

    elif isinstance(bins, np.ndarray):
        # Check that bins are well defined (monotonically increasing)
        assert np.all(np.diff(bins) > 0), 'Bin definition is ill-formed.'

    # Check that given bin range matches the data values
    if values is not None and values.size > 0:
        if check_range:
            check_bin_range(values, bins)

    return bins
