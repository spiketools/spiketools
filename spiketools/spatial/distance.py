"""Distance related functions."""

import numpy as np

from spiketools.utils.checks import check_array_orientation
from spiketools.utils.data import make_orientation

###################################################################################################
###################################################################################################

def compute_distance(p1, p2):
    """Compute the distance between two positions.

    Parameters
    ----------
    p1, p2 : list of float
        The position values of the two positions to calculate distance between.
        Can be 1d (a single value per position) or 2d (x and y values per position).

    Returns
    -------
    float
        Distance between the two positions.

    Examples
    --------
    Compute distance between two 1d positions:

    >>> p1, p2 = [2], [5]
    >>> compute_distance(p1, p2)
    3.0

    Compute distance between the two 2d positions:

    >>> p1 = [1, 6]
    >>> p2 = [5, 9]
    >>> compute_distance(p1, p2)
    5.0
    """

    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_distances(position):
    """Compute distances across a sequence of positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.

    Returns
    -------
    distances : 1d array
        Vector of distances between positions.

    Examples
    --------
    Compute distances across a sequence of 1d positions:

    >>> position = np.array([1., 2., 4., 5.])
    >>> compute_distances(position)
    array([1., 2., 1.])

    Compute distances across a sequence of 2d positions:

    >>> position = np.array([[1, 2, 2, 3],
    ...                      [1, 1, 2, 3]])
    >>> compute_distances(position)
    array([1.        , 1.        , 1.41421356])
    """

    position = make_orientation(position, 'column')

    distances = np.zeros(len(position) - 1)
    for ix, (p1, p2) in enumerate(zip(position, position[1:])):
        distances[ix] = compute_distance(p1, p2)

    return distances


def compute_cumulative_distances(position, align_output=True):
    """Compute cumulative distance across a sequence of positions.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    align_output : bool, optional, default: True
        If True, aligns the output with the sampling of the input, to match length.
        To do so, value of 0 is prepended to the output array.

    Returns
    -------
    1d array
        Cumulative distances.

    Examples
    --------
    Compute cumulative distances across a sequence of 1d positions:

    >>> position = np.array([1., 2., 4., 5.])
    >>> compute_cumulative_distances(position)
    array([0., 1., 3., 4.])

    Compute cumulative distances across a sequence of 2d positions:

    >>> position = np.array([[1, 2, 2, 3],
    ...                      [1, 1, 2, 3]])
    >>> compute_cumulative_distances(position)
    array([0.        , 1.        , 2.        , 3.41421356])
    """

    cumul_dists = np.cumsum(compute_distances(position))

    if align_output:
        cumul_dists = np.insert(cumul_dists, 0, 0)

    return cumul_dists


def compute_distances_to_location(position, location):
    """Compute distances between a sequence of positions and a specified location.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    location : list of float
        The position values of the two positions to calculate distance between.
        Can be 1d (a single value per position) or 2d (x and y values per position).

    Returns
    -------
    distances : 1d array
        Computed distances between each position in the sequence, and the specified location.

    Examples
    --------
    Compute distances to location across a sequence of 1d positions:

    >>> position = np.array([1., 2., 3., 4., 5.])
    >>> location = [3.]
    >>> compute_distances_to_location(position, location)
    array([2., 1., 0., 1., 2.])

    Compute distances to location across a sequence of 2d positions:

    >>> position = np.array([[1, 2, 2, 3],
    ...                      [1, 1, 2, 3]])
    >>> location = [2, 2]
    >>> compute_distances_to_location(position, location)
    array([1.41421356, 1.        , 0.        , 1.41421356])
    """

    position = make_orientation(position, 'column')

    distances = np.zeros(len(position))
    for ix, pos in enumerate(position):
        distances[ix] = compute_distance(pos, location)

    return distances


def get_closest_position(position, location, threshold=None):
    """Get the index of the closest position value to a specified location.

    Parameters
    ----------
    position : 1d or 2d array
        Position values.
    location : list of float
        The position values of the two positions to calculate distance between.
        Can be 1d (a single value per position) or 2d (x and y values per position).
    threshold : float, optional
        The threshold that the closest value must be within to be returned.
        If the distance is greater than the threshold, output is -1.

    Returns
    -------
    min_ind : int
        The index of the closest position value to the specified location.

    Notes
    -----
    If there are multiple positions that are equivalently close to the location,
    this function will return the index of first one.

    Examples
    --------
    Get the index of the closest 1d position to a specified location:

    >>> position = np.array([1., 2., 3., 4., 5.])
    >>> location = [3.]
    >>> get_closest_position(position, location)
    2

    Get the index of the closest 2d position to a specified location:

    >>> position = np.array([[1, 2, 2, 3],
    ...                      [1, 1, 2, 3]])
    >>> location = [2, 2]
    >>> get_closest_position(position, location)
    2
    """

    distances = compute_distances_to_location(position, location)

    min_ind = np.argmin(distances)

    if threshold:
        if distances[min_ind] > threshold:
            min_ind = -1

    return min_ind
