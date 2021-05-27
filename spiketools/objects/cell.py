"""Cell object."""

import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

class Cell(object):
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

        spike_train = np.zeros(np.ceil(self.times[-1]).astype(int))
        ts_ind = [int(i) for i in self.times if i < spike_train.shape[-1]]
        spike_train[ts_ind] = 1

        return spike_train


    def ISI(self, plot=False):
        """Compute and plot the ISI."""

        ISI = np.diff(self.times)

        if plot:
            f, ISI_hist = plt.subplots(1, 1, figsize=(10, 4))
            ISI_hist.hist(ISI)

        return ISI


    def CV(self):
        """Compute coefficient of variation."""

        ISI = self.ISI()
        CV = np.std(ISI) / np.mean(ISI)

        return CV


    def fano(self):
        """Compute fano factor."""

        spiketrain = self.spike_train()
        fano = np.var(spiketrain) / np.mean(spiketrain)

        return fano


    def ISI_shuffle(self, random_state=None):
        """Shuffle the ISI and return new spike times."""

        # initialize empty array
        st = np.zeros_like(self.times)

        rng = np.random.RandomState(random_state)

        ISI = np.diff(self.times)

        st[1:] = np.cumsum(rng.permutation(ISI)) + st[0]

        return st


    @staticmethod
    def poisson_train(frequency, duration, start_time=0, seed=None):
        """Generator function for a Homogeneous Poisson train.

        Parameters
        ----------
        frequency : xx
            The mean spiking frequency.
        duration :
            Maximum duration.
        start_time:
            Timestamp.
        seed :
            Seed for the random number generator.
            If None, this will be decided by np, which chooses the system time.

        Yields
        ------
        val
            A relative spike time from t=start_time, in seconds (not ms).

        EXAMPLE::

            # Make a list of spikes at 20 Hz for 3 seconds
            spikes = [i for i in poisson_train(20, 3)]

        EXAMPLE::

            # Use dynamically in a program
            # Care needs to be taken with this scenario because the generator will
            # generate spikes until the program or spike_gen object is terminated.
            spike_gen = poisson_train(20, duration=sys.float_info.max)
            spike = spike_gen.next()
            # Process the spike, to other programmatic things
            spike = spike_gen.next() # Get another spike
            # etc.
            # Terminate the program.
        """

        cur_time = start_time
        rangen = np.random.mtrand.RandomState()

        if seed is not None:
            rangen.seed(seed)

        isi = 1. / frequency

        while cur_time <= duration:

            cur_time += isi * rangen.exponential()

            if cur_time > duration:
                return

            yield cur_time
