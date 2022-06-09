"""Session objects."""

###################################################################################################
###################################################################################################

class Session():
    """A Session object.

    Parameters
    ----------
    subject : str
        Subject label.
    session : str
        Session label.
    task : str
        Task label.
    units : list of Units
        Units for the session.
    """

    def __init__(self, subject=None, session=None, task=None, units=None):
        """Initialize Subject object."""

        self.subject = subject
        self.session = session
        self.task = task
        self.units = units if units is not None else []


    def add_unit(self, unit):
        """Add a unit to the object.

        Parameters
        ----------
        unit : Unit
            An individual unit.
        """

        self.units.append(unit)


    @property
    def n_units(self):
        """The number of units contained in the object."""

        return len(self.units)
