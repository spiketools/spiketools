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
    """

    base = '{:1.2f} hours, {:1.2f} minutes, and {:1.2f} seconds.'
    return base.format(hours, minutes, seconds)
