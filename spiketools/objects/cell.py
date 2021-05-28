"""Cell object."""

import numpy as np

from spiketools.stats.shuffle import shuffle_isis
from spiketools.measures import create_spike_train
from spiketools.measures import compute_isis, compute_cv, compute_fano_factor

###################################################################################################
###################################################################################################

class Cell():
    """A Session object.

    Parameters
    ----------
    subject : str
        Subject label.
    session : str
        Session label.
    task : str
        Task label.
    channel : str
        Channel label.
    region : str
        Region label.
    """

    def __init__(self, subject=None, session=None, task=None, channel=None, region=None):
        """Initialize a Cell object."""

        self.subject = subject
        self.session = session
        self.task = task
        self.channel = channel
        self.region = region

        # NOTE: need to update to add / load spike times and cluster information
        self.times = None
        self.cluster = None


    def spike_train(self):
        """Convert spike times into a spike train vector (binary)."""

        return create_spike_train(self.times)


    def ISI(self):
        """Compute and plot the ISI."""

        return compute_isis(self.times)


    def CV(self):
        """Compute coefficient of variation."""

        return compute_cv(self.ISI())


    def fano(self):
        """Compute fano factor."""

        return compute_fano_factor(self.spike_train())


    def ISI_shuffle(self, random_state=None):
        """Shuffle the ISI and return new spike times."""

        return shuffle_isis(self.times, random_state=random_state)
