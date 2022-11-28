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


def get_function_argument(label, function, args, kwargs, argind=-1):
    """Get a function argument value from it's inputs or signature.

    Parameters
    ----------
    label : str
        Name of the argument to access.
    function : callable
        Function.
    args : tuple
        Arguments.
    kwargs : dict
        Keyword arguments.
    argind : int, optional, default=-1
        Index of arguments that label should be at, if defined there.

    Returns
    -------
    output
        The accessed argument value.
    """

    # Define a 'null' output (safer in case real value is "None", "False", etc)
    output = 'null'

    # Try and access the requested label from kwargs
    try:
        output = kwargs.pop(label)
    except KeyError:
        pass

    func_params = get_function_parameters(function)

    # If output not yet defined, check args for
    if output == 'null':
        n_args = len(func_params)
        argind = n_args if argind == -1 else argind
        try:
            output = args[argind]
        except IndexError:
            pass

    # If output still not defined, get default value from the function signature
    if output == 'null':
        try:
            output = func_params[label].default
        except KeyError:
            # If nothing so far, return as None
            output = None

    return output
