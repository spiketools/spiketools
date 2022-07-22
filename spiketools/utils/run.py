"""Utilities for managing analysis runs."""

###################################################################################################
###################################################################################################

def create_methods_list(methods):
    """Create a list of all methods.

    Parameters
    ----------
    methods : dict
        Analysis method(s) definition.
        Each key should be an analysis name, and each value a list of methods to run.

    Returns
    -------
    methods_lst : list of str
        A list of all defined methods, each one as 'analysis_method'.
    """

    methods_lst = []
    for analysis, method in methods.items():
        for temp in method:
            methods_lst.append(analysis.lower() + '_' + temp.lower())

    return methods_lst
