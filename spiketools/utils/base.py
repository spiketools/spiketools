"""Base utility functions, that manipulate basic data structures, etc."""

from collections import Counter
from collections.abc import Iterable

import numpy as np

###################################################################################################
###################################################################################################

def flatten(lst):
    """Flatten a list of lists into a single list.

    Parameters
    ----------
    lst : list of list
        A list of embedded lists.

    Returns
    -------
    lst
        A flattened list.

    Examples
    --------
    Flatten a list containing three embedded lists:

    >>> lst = [[1, 2, 3, 4], [5, 6, 7 ,8], [9, 10, 11, 12]]
    >>> flatten(lst)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    """

    return [item for sublist in lst for item in sublist]


def lower_list(lst):
    """Convert all strings in a list to lowercase.

    Parameters
    ----------
    lst : list of str
        A list of strings.

    Returns
    -------
    lst
        A list of string, all lowercase.

    Examples
    --------
    Lower a list of mixed case entries:

    >>> lst = ['First', 'SECOND', 'Third']
    >>> lower_list(lst)
    ['first', 'second', 'third']
    """

    return [el.lower() for el in lst]


def select_from_list(lst, select):
    """Select elements from a list based on a boolean mask.

    Parameters
    ----------
    lst : list
        A list of values to select from.
    select : list of bool
        Indicator for which elements to select.

    Returns
    -------
    lst
        Selected elements from the list.

    Examples
    --------
    Select the first and last element from given list:

    >>> lst = [1, 4, 3, 6, 8]
    >>> select = [True, False, False, False, True]
    >>> select_from_list(lst, select)
    [1, 8]
    """

    return [el for el, sel in zip(lst, select) if sel]


def count_elements(data, labels=None, sort=False):
    """Count elements within a collection object.

    Parameters
    ----------
    data : list or 1d array
        The data to count elements of.
    labels : list of str or 'count', optional
        Labels to ensure are in the counter, adding a count of zero if missing.
        If list, should be a list of labels to check.
        If 'count', then adds labels for integer count values across the observed range.
    sort : bool, optional, default: False
        Whether to sort the counter, by key, before returning.

    Returns
    -------
    counts : Counter
        Counts of the elements within the given data object.

    Examples
    --------
    Count the number of occurrences of each element in a 1d array:

    >>> data = [1, 3, 3, 4, 5, 6, 9, 3, 4, 5, 6]
    >>> count_elements(data)
    Counter({3: 3, 4: 2, 5: 2, 6: 2, 1: 1, 9: 1})
    """

    counts = Counter(data)

    if labels is not None:
        if labels == 'count':
            labels = range(0, max(list(counts.keys())))
        for label in labels:
            counts.setdefault(label, 0)

    if sort:
        counts = Counter(dict(sorted(counts.items())))

    return counts


def combine_dicts(dicts):
    """Combine a list of dictionaries together.

    Parameters
    ----------
    dicts : list of dict
        Dictionaries to combine.

    Returns
    -------
    dict
        Combined dictionary.

    Notes
    -----
    If multiple dictionaries have the same keys, the value of the last dictionary is kept.

    Examples
    --------
    Combine two dictionaries:

    >>> dict1 = {'a' : 1, 'b' : 2}
    >>> dict2 = {'c' : 3, 'd' : 4}
    >>> combine_dicts([dict1, dict2])
    {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    """

    output = {}
    for cdict in dicts:
        output.update(cdict)

    return output


def add_key_prefix(indict, prefix):
    """Update keys of a dictionary by appending a prefix.

    Parameters
    ----------
    indict : dict
        Dictionary to update keys for.
    prefix : str
        Prefix to add to each dictionary key.

    Returns
    -------
    dict
        Dictionary with updated keys.

    Examples
    --------
    Add a prefix to dictionary keys:

    >>> indict = {'setting' : 12, 'param' : 22}
    >>> add_key_prefix(indict, 'analysis')
    {'analysis_setting': 12, 'analysis_param': 22}
    """

    out = {}
    for key, value in indict.items():
        out[prefix + '_' + key] = value

    return out


def drop_key_prefix(indict, prefix):
    """Update keys of a dictionary by dropping a prefix.

    Parameters
    ----------
    indict : dict
        Dictionary to update keys for.
    prefix : str
        Prefix to drop from each dictionary key.

    Returns
    -------
    dict
        Dictionary with updated keys.

    Examples
    --------
    Drop a prefix from dictionary keys:

    >>> indict = {'analysis_setting' : 12, 'analysis_param' : 22}
    >>> drop_key_prefix(indict, 'analysis')
    {'setting': 12, 'param': 22}
    """

    out = {}
    for key, value in indict.items():
        key_elements = key.split('_')
        if prefix in key_elements:
            key_elements.remove(prefix)
        out['_'.join(key_elements)] = value

    return out


def relabel_keys(indict, new_keys):
    """Relabel keys in a dictionary.

    Parameters
    ----------
    indict : dict
        Dictionary with key names to be updated.
    new_keys : dict
        Dictionary defining new key names.
        Each key should be the current name, and each value the name to update to.

    Returns
    -------
    outdict : dict
        Dictionary with updated key names.

    Examples
    --------
    Relabel a set of keys in a dictionary:

    >>> dictionary = {'spike_name' : 'a1', 'spike_type' : 0}
    >>> new_keys = {'spike_name' : 'name', 'spike_type' : 'type'}
    >>> relabel_keys(dictionary, new_keys)
    {'name': 'a1', 'type': 0}
    """

    outdict = {}
    for key in indict:
        outdict[new_keys.get(key, key)] = indict[key]

    return outdict


def subset_dict(indict, label):
    """Subset a dictionary based on key labels.

    Parameters
    ----------
    indict : dict
        Dictionary to subset.
    label : str
        Label to use to subset keys.

    Returns
    -------
    dict
        Subsetted dictionary.

    Examples
    --------
    Subset a set of specified keys from a dictionary:

    >>> dictionary = {'spike_name' : 'a1', 'spike_type' : 0, 'data' : [1, 2, 3]}
    >>> label = 'spike'
    >>> subset_dict(dictionary, label)
    {'spike_name': 'a1', 'spike_type': 0}
    """

    output = {}
    for key in list(indict.keys()):
        if label in key:
            output[key] = indict.pop(key)

    return output


def check_keys(indict, lst):
    """Check a dictionary for a set of keys.

    Parameters
    ----------
    indict : dict
        Dictionary to check keys of.
    lst : list
        List of keys to check in the dictionary.

    Returns
    -------
    str or None
        Key label that was found in the dictionary, or None if no specified keys found.

    Notes
    -----
    If more than one of the specified keys are in the dictionary, the first one is returned.

    Examples
    --------
    Check which key is defined in dictionary:

    >>> dictionary = {'spike_name' : 'a1', 'spike_type' : 0}
    >>> lst = ['spike_name', 'spike_label']
    >>> check_keys(dictionary, lst)
    'spike_name'
    """

    for el in lst:
        if el in indict.keys():
            output = el
            break
    else:
        output = None

    return output


def listify(param, index=False):
    """Check and embed a parameter into a list, if is not already in a list.

    Parameters
    ----------
    param : object
        Parameter to check and embed in a list, if it is not already.
    index : bool, optional
        If True, indexes into `param` to check the 0th element, instead of `param` itself.
        This can be used for checking and embedding a list into a list.

    Returns
    -------
    list
        Parameter embedded in a list.
    """

    check = param[0] if index else param

    # Embed all non-iterable parameters into a list
    #   Note: deal with str as a special case of iterable that we want to embed
    if not isinstance(check, Iterable) or isinstance(check, str):
        out = [param]
    # Deal with special case of multi dimensional numpy arrays - want to embed without flattening
    elif isinstance(check, np.ndarray) and np.ndim(check) > 1:
        out = [param]
    # If is iterable (e.g. tuple or numpy array), typecast to list
    else:
        out = list(param)

    return out
