"""Module level utilities for working with functions."""

from inspect import signature

###################################################################################################
###################################################################################################

def get_function_parameters(function):
    """Get a list of input parameters for a function.

    Parameters
    ----------
    function : callable
        A function to get the parameters from.

    Returns
    -------
    parameters : list of str
        Names of the input parameters to the given function.
    """

    sig = signature(function)
    parameters = list(sig.parameters.keys())

    return parameters
