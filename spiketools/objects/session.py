"""Session objects."""

###################################################################################################
###################################################################################################

class Session(object):
    """A Session object.

    Parameters
    ----------
    subject : str
        Suject label.
    session : str
        Session label.
    task : str
        Task label.
    """

    def __init__(self, subject=None, session=None, task=None):
        """Initialize Subject object."""

        self.subject = subject
        self.session = session
        self.task = task
