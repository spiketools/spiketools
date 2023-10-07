"""General purpose checker functions."""

import warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np

from spiketools.utils.base import lower_list, listify

###################################################################################################
###################################################################################################

AXISARG = defaultdict(lambda : -1, {'vector' : 0, 'row' : 1, 'column' : 0})

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


def check_param_type(param, label, types):
    """Check that a parameter has an acceptable type.

    Parameters
    ----------
    param : object
        Parameter to check type of.
    label : str
        Label of the parameter being checked.
    types : type or tuple of type
        Type(s) to check the given parameter is one of.

    Raises
    ------
    ValueError
        If the parameter is not one of the specified acceptable types.

    Notes
    -----
    If checking elements in an array, then proper checking should include relevant numpy types.
    """

    types = tuple(listify(types)) if not isinstance(types, tuple) else types
    if not isinstance(param, types):
        type_strings = [str(el).split(' ')[1][1:-2] for el in types]
        msg = "The parameter {} should have type: {}.".format(label, type_strings)
        raise TypeError(msg)


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


def check_array_orientation(arr, expected=None):
    """Check the orientation of an array of data.

    Parameters
    ----------
    arr : ndarray
        Data array to check the orientation of.
    expected : int, optional
        The expected number of values per sample (e.g. 2 for (x,y) position values).
        If provided, is used to infer orientation in ambiguous cases (e.g. # samples < # values).

    Returns
    -------
    orientation : {'vector', 'row', 'column'}
        The inferred orientation of the data array.
        For 1d arrays, 'vector' is returned.
        For 2d or 3rd arrays, 'row' or 'column' is returned based on the shape of the array.

    Notes
    -----
    In cases where # elements > 0 <= # dimensions, orientation can be ambiguous.
    In such cases, if `expected` is not provided, 'row' is returned by default.
    """

    assert arr.ndim < 4, "The `check_array_orientation` function only works up to 3d."

    if arr.ndim == 1:
        orientation = 'vector'

    else:

        shape = np.array(arr.shape)

        # If expected number of values is given, use this
        if expected is not None and sum(shape == expected) == 1:
            opts = ['trial', 'row', 'column'] if arr.ndim == 3 else ['row', 'column']
            orientation = opts[np.where(shape == expected)[0][0]]

        # Special case - empty array, infer based on where zero dimension is
        elif sum(shape == 0) == 1:
            opts = ['trial', 'column', 'row'] if arr.ndim == 3 else ['column', 'row']
            orientation = opts[np.where(shape == 0)[0][0]]

        # Otherwise, infer shape based on the relative size of each dimension
        else:
            if shape[-1] >= shape[-2]:
                orientation = 'row'
            else:
                orientation = 'column'

    return orientation


def check_array_lst_orientation(arr_lst, expected=None):
    """Check the orientation of arrays in a list.

    Parameters
    ----------
    arr_lst : list of array
        List of arrays to check orientation for.
    expected : int, optional
        The expected number of values per sample (e.g. 2 for (x,y) position values).
        If provided, is used to infer orientation in ambiguous cases (e.g. # samples < # values).

    Returns
    -------
    orientation : {'vector', 'row', 'column'}
        The inferred orientation of the data array.
        For 1d arrays, 'vector' is returned.
        For 2d or 3d arrays, 'row' or 'column' is returned based on the shape of the array.
    """

    # Special case - empty list: return default option
    if len(arr_lst) == 0:
        orientation = None

    else:
        # Find an array with enough elements to infer orientation
        array = arr_lst[0]
        for cur_arr in arr_lst:
            if cur_arr.size > 4:
                array = cur_arr
                break

        orientation = check_array_orientation(array, expected)

    return orientation


def check_axis(axis, arr):
    """Check axis argument, and infer from array orientation if not defined.

    Parameters
    ----------
    axis : {None, 0, 1, -1}
        Axis argument.
        If not None, this value is returned.
        If None, the given array is checked to infer axis.
    arr : ndarray or list of ndarray
        Array to check the axis argument for.

    Returns
    -------
    axis : {0, 1, -1}
        Axis argument.
        For 1d array, 0 is returned, reflecting a vector.
        For 2d array, 0 is for column, and 1 is for row.
        If the axis could not be inferred, -1 is returned.
    """

    if not axis:

        if isinstance(arr, list):
            orientation = check_array_lst_orientation(arr)
        else:
            orientation = check_array_orientation(arr)

        axis = AXISARG[orientation]

    return axis


def check_bin_range(values, bin_area):
    """Checks data values against given bin edges, warning if values exceed bin range.

    Parameters
    ----------
    values : 1d array
        A set of values to check against bin edges.
    bin_area : 1d array or list
        The bin range area to check. Can be a two-item area range, or an array of bin edges.
    """

    if values.size > 0:
        if np.nanmin(values) < bin_area[0] or np.nanmax(values) > bin_area[-1]:
            msg = 'The data values extend beyond the given bin definition.'
            warnings.warn(msg)


def check_time_bins(bins, time_range=None, values=None, check_range=False):
    """Check a given time bin definition, and define if only given a time resolution.

    Parameters
    ----------
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    time_range : list of [float, float], optional
        Time range, in seconds, to create the binned firing rate across.
        Only used if `bins` is a float. If given, the end value is inclusive.
    values : 1d array, optional
        The time values that are to be binned.
        Optional if `time_range` is provided instead.
    check_range : bool, optional, default: False
        Whether to check the range of the data values against the time bins.

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
    >>> check_time_bins(bins, time_range, values)
    array([0. , 0.5, 1. , 1.5, 2. ])

    Check a time bin definition, where bins are already defined:

    >>> bins = np.array([0. , 0.5, 1. , 1.5, 2. ])
    >>> check_time_bins(bins, time_range, values)
    array([0. , 0.5, 1. , 1.5, 2. ])
    """

    # Take a copy of `time_range` (otherwise, can get an aliasing problem)
    time_range = deepcopy(time_range)

    check_param_type(bins, 'bins', (int, float, np.ndarray))

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
