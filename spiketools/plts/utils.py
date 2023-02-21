"""Plot utilities."""

import math
from functools import wraps
from os.path import join as pjoin

import matplotlib.pyplot as plt
from matplotlib import gridspec

###################################################################################################
###################################################################################################

def check_ax(ax, figsize=None, return_current=False):
    """Check whether a figure axes object is defined, and define or return current axis if not.

    Parameters
    ----------
    ax : matplotlib.Axes or None
        Axes object to check if is defined.
    figsize : tuple of float, optional
        Size to make the axis.
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


def get_kwargs(kwargs, select):
    """Get keyword arguments.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to extract from.
    select : list of str
        The arguments to extract.

    Returns
    -------
    setters : dict
        Selected keyword arguments.
    """

    setters = {arg : kwargs.pop(arg, None) for arg in select}
    setters = {arg : value for arg, value in setters.items() if value is not None}

    return setters


def get_attr_kwargs(kwargs, attr):
    """Get keyword arguments related to a particular attribute.

    Parameters
    ----------
    kwargs : dict
        Plotting related keyword arguments.
    attr : str
        The attribute to select related arguments.

    Returns
    -------
    attr_kwargs : dict
        Selected keyword arguments, related to the given attribute.
    """

    labels = [key for key in kwargs.keys() if attr in key]
    attr_kwargs = {label.split('_')[1] : kwargs.pop(label) for label in labels}

    return attr_kwargs


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

        # Check and collect any other plot keywords
        save_kwargs = kwargs.pop('save_kwargs', {})
        save_kwargs.setdefault('bbox_inches', 'tight')

        # Check and collect whether to close the plot
        close = kwargs.pop('close', False)

        func(*args, **kwargs)

        if save_fig:
            save_figure(file_name, file_path, close, **save_kwargs)

    return decorated


def save_figure(file_name, file_path=None, close=False, **save_kwargs):
    """Save out a figure.

    Parameters
    ----------
    file_name : str
        File name for the figure file to save out.
    file_path : str or Path
        Path for where to save out the figure to.
    close : bool, optional, default: False
        Whether to close the plot after saving.
    save_kwargs
        Additional arguments to pass into the save function.
    """

    full_path = pjoin(file_path, file_name) if file_path else file_name
    plt.savefig(full_path, **save_kwargs)

    if close:
        plt.close()


def make_axes(n_axes, n_cols=5, figsize=None, row_size=4, col_size=3.6,
              wspace=None, hspace=None, title=None, **plt_kwargs):
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
    title : str, optional
        A title to add to the figure.
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

    title_kwargs = get_attr_kwargs(plt_kwargs, 'title')

    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize, **plt_kwargs)

    if wspace or hspace:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Turn off axes for any extra subplots in last row
    _ = [ax.axis('off') for ax in axes.ravel()[n_axes:]]

    if title:
        plt.suptitle(title,
                     fontsize=title_kwargs.pop('title_fontsize', 24),
                     **title_kwargs)

    return axes.flatten()


def make_grid(nrows, ncols, title=None, **plt_kwargs):
    """Create a plot grid.

    Parameters
    ----------
    nrows, ncols : int
        The number of rows and columns to add to the data.
    title : str, optional
        A title to add to the figure.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    title_kwargs = get_attr_kwargs(plt_kwargs, 'title')

    _ = plt.figure(figsize=plt_kwargs.pop('figsize', None))
    grid = gridspec.GridSpec(nrows, ncols, **plt_kwargs)

    if title:
        plt.suptitle(title,
                     fontsize=title_kwargs.pop('title_fontsize', 24),
                     y=title_kwargs.pop('title_y', 0.95),
                     **title_kwargs)

    return grid


def get_grid_subplot(grid, row, col, **plt_kwargs):
    """Get a subplot section from a grid layout.

    Parameters
    ----------
    grid : matplotlib.gridspec.GridSpec
        A predefined plot grid layout.
    row, col : int or slice
        The row(s) and column(s) in which to place the subplot.
    plt_kwargs
        Additional arguments to pass into the plot function.

    Returns
    -------
    matplotlib.AxesSubplot
        Subplot axis.
    """

    return plt.subplot(grid[row, col], **plt_kwargs)
