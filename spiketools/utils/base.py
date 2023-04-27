"""Base utility functions, that manipulate basic data structures, etc."""

from collections import Counter

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
    Flatten a list of 3 lists inside:

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
        Prefix to add to each dictionary.

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
