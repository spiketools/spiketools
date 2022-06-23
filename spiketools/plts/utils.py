"""Plot utilities."""

import math
from functools import wraps
from os.path import join as pjoin

import matplotlib.pyplot as plt

from spiketools.plts.settings import SET_KWARGS

###################################################################################################
###################################################################################################

def check_ax(ax, figsize=None, return_current=False):
    """Check whether a figure axes object is defined, and define or return current axis if not.

    Parameters
    ----------
    ax : matplotlib.Axes or None
        Axes object to check if is defined.
    return_current : bool, optional, default: False
        Whether to return the current axis, if axis is not defined.
        If False, creates a new plot axis instead.

    Returns
    -------
    ax : matplotlib.Axes
        Figure axes object to use.
    """

    if not ax:
        if return_current:
            ax = plt.gca()
        else:
            _, ax = plt.subplots(figsize=figsize)

    return ax


def savefig(func):
    """Decorator function to save out figures."""

    @wraps(func)
    def decorated(*args, **kwargs):

        # Grab file name and path arguments, if they are in kwargs
        file_name = kwargs.pop('file_name', None)
        file_path = kwargs.pop('file_path', None)

        # Check for an explicit argument for whether to save figure or not
        #   Defaults to saving when file name given (since bool(str)->True; bool(None)->False)
        save_fig = kwargs.pop('save_fig', bool(file_name))

        # Check any collect any other plot keywords
        save_kwargs = kwargs.pop('save_kwargs', {})
        save_kwargs.setdefault('bbox_inches', 'tight')

        # Check and collect whether to close the plot
        close = kwargs.pop('close', None)

        func(*args, **kwargs)

        if save_fig:
            full_path = pjoin(file_path, file_name) if file_path else file_name
            plt.savefig(full_path, **save_kwargs)

        if close:
            plt.close()

    return decorated


def set_plt_kwargs(func):
    """Collects and then sets plot kwargs that can be applied with 'set'."""

    @wraps(func)
    def decorated(*args, **kwargs):

        setters = {arg : kwargs.pop(arg, None) for arg in SET_KWARGS}
        setters = {arg : value for arg, value in setters.items() if value}

        func(*args, **kwargs)

        ax = kwargs['ax'] if 'ax' in kwargs and kwargs['ax'] is not None else plt.gca()
        ax.set(**setters)

    return decorated


def make_axes(n_axes, n_cols=5, figsize=None, row_size=4, col_size=3.6,
              wspace=None, hspace=None, **plt_kwargs):
    """Make a subplot with multiple axes.

    Parameters
    ----------
    n_axes : int
        The total number of axes to create in the figure.
    n_cols : int, optional, default: 5
        The number of columns in the figure.
    figsize : tuple of float, optional
        Size to make the overall figure.
        If not given, is estimated from the number of axes.
    row_size, col_size : float, optional
        The size to use per row / column.
        Only used if `figsize` is None.
    wspace, hspace : float, optional
        Spacing parameters for between subplots.
        These get passed into `plt.subplots_adjust`.
    **plt_kwargs
        Extra arguments to pass to `plt.subplots`.

    Returns
    -------
    axes : 1d array of AxesSubplot
        Collection of axes objects.
    """

    n_rows = math.ceil(n_axes / n_cols)

    if not figsize:
        figsize = (n_cols * col_size, n_rows * row_size)

    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize, **plt_kwargs)

    if wspace or hspace:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Turn off axes for any extra subplots in last row
    [ax.axis('off') for ax in axes.ravel()[n_axes:]]

    return axes.flatten()
