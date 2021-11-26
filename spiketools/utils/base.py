"""Base utility functions, that manipulate basic data structures, etc."""

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
    """

    return [item for sublist in lst for item in sublist]
