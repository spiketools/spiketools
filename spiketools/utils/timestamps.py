"""Utility functions for working with timestamps."""

###################################################################################################
###################################################################################################

def convert_ms_to_sec(ms):
    """Convert time value(s) from milliseconds to seconds.

    Parameters
    ----------
    ms : float
        Time value(s), in milliseconds.

    Returns
    -------
    float or array
        Time value(s), in seconds.

    Examples
    --------
    Convert milliseconds to seconds:

    >>> convert_ms_to_sec(500)
    0.5
    """

    return ms / 1000


def convert_sec_to_min(sec):
    """Convert time value(s) from seconds to minutes.

    Parameters
    ----------
    sec : float or array
        Time value(s), in seconds.

    Returns
    -------
    float or array
        Time value(s), in minutes.

    Examples
    --------
    Convert seconds to minutes:

    >>> convert_sec_to_min(210)
    3.5
    """

    return sec / 60


def convert_min_to_hour(mins):
    """Convert time value(s) from minutes to hours.

    Parameters
    ----------
    ms : float or array
        Time value(s), in minutes.

    Returns
    -------
    float or array
        Time value(s), in hours.

    Examples
    --------
    Convert minutes to hours:

    >>> convert_min_to_hour(270)
    4.5
    """

    return mins / 60


def convert_ms_to_min(ms):
    """Convert time value(s) from milliseconds to minutes.

    Parameters
    ----------
    ms : float or array
        Time value(s), in milliseconds.

    Returns
    -------
    float or array
        Time value(s), in minutes.

    Examples
    --------
    Convert milliseconds to minutes:

    >>> convert_ms_to_min(150000)
    2.5
    """

    return convert_sec_to_min(convert_ms_to_sec(ms))


def split_time_value(sec):
    """Split a time value from seconds to hours / minutes / seconds.

    Parameters
    ----------
    sec : float
        Time value, in seconds.

    Returns
    -------
    hours, minutes, seconds : float
        Time value, split up into hours, minutes, and seconds.

    Examples
    --------
    Split seconds into hours, minutes, and seconds:

    >>> split_time_value(15000)
    (4, 10, 0)
    """

    minutes, seconds = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)

    return hours, minutes, seconds


def format_time_string(hours, minutes, seconds):
    """Format a time value into a string.

    Parameters
    ----------
    hours, minutes, seconds : float
        Time value, represented as hours, minutes, and seconds.

    Returns
    -------
    str
        A string representation of the time value.

    Examples
    --------
    Format a time stored as hours, minutes, and seconds into a string:

    >>> format_time_string(4, 10, 20)
    '4.00 hours, 10.00 minutes, and 20.00 seconds.'
    """

    base = '{:1.2f} hours, {:1.2f} minutes, and {:1.2f} seconds.'
    return base.format(hours, minutes, seconds)
