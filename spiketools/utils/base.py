"""Base utility functions, that manipulate basic data structures, etc."""

from collections import Counter
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
    Flatten a list of 3 lists inside. 

    >>> lst = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    >>> flatten(lst)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    """

    return [item for sublist in lst for item in sublist]


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
    Select the first and last element from given list. 

    >>> lst = [1,4,3,6,8]
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
    Count the number of occurance of each element in a 1d array.

    >>> data = np.array([1,3,3,4,5,6,9,3,4,5,6])
    >>> count_elements(data)
    Counter({1: 1, 3: 3, 4: 2, 5: 2, 6: 2, 9: 1})
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
