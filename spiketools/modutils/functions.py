"""Module level utilities for working with functions."""

from inspect import signature

###################################################################################################
###################################################################################################

def get_function_parameters(function):
    """Get the input parameters for a function.

    Parameters
    ----------
    function : callable
        A function to get the parameters from.

    Returns
    -------
    parameters : dict
        Parameter definition of the given function.
        Each key is a str label of the parameter name.
        Each value is a inspect.Parameter object describing the parameter.
    """

    func_signature = signature(function)
    parameters = dict(func_signature.parameters)

    return parameters
