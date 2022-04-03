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
