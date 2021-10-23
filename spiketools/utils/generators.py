"""Generator utility functions.."""

import inspect

###################################################################################################
###################################################################################################

def incrementer(start=0, end=999):
    """Generator that returns an incrementing index value.

    Parameters
    ----------
    start, end : int
        The start and end point for the incrementer.

    Yields
    ------
    ind : int
        The current index value.
    """

    for ind in range(start, end):
        yield ind
