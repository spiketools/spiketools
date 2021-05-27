import h5py
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from glob import glob


class Cell(object):
    def __init__(self, subj=None, sess=None, task=None, chan=None, region=None):
        """
        A cell object that will have a time, cluster, subject, region, etc.
        and have methods for computing spatial metrics
        """

        self.subject = subj
        self.session = sess
        self.channel = chan
        self.region = region
        self.task = task
        all_paths = glob(f'/home1/salman.qasim/T3_Data/Subjects/{subj}/session_{sess}/split_files/times*')
        if all_paths[0].split('.')[-1] == 'h5':
            chan_path = [x for x in all_paths if chan == int(x.split('/')[-1].split('_')[-1][4:-3])]
        elif all_paths[0].split('.')[-1] == 'mat':
            chan_path = [x for x in all_paths if chan == int(x.split('/')[-1].split('_')[-1][4:-4])]
        self.path = chan_path[0]
        self.times = None
        self.cluster = None

    def _load_data(self):
        """
        Load the data as a cell attribute
        """

        if self.path.split('.')[-1] == 'h5':
            # For use with the T3 data
            f = h5py.File(self.path, 'r')
            data = f.get(list(f.keys())[0])
            self.times = np.array(data.get('spikeTimes'))
            self.cluster = np.array(data.get('spikeClusters'))

        elif self.path.split('.')[-1] == 'mat':
            # For use with the Circ, Train data
            data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)['events']
            self.times = np.array(data['spikeData']['spikeTimes'])
            self.cluster = np.array(data['spikeData']['spikeNames'])

    def spike_train(self):
        """
        turn spike times into a spike train vector (binary)
        """

        spike_train = np.zeros(np.ceil(self.times[-1]).astype(int))
        ts_ind = [int(i) for i in self.times if i < spike_train.shape[-1]]
        spike_train[ts_ind] = 1

        return spike_train

    def ISI(self, plot=False):
        """
        Quickly compute and plot the ISI
        :param plot:
        :param st: array of spiketimes in milliseconds
        :return:
        """

        ISI = np.diff(self.times)
        if plot:
            f, ISI_hist = plt.subplots(1, 1, figsize=(10, 4))
            ISI_hist.hist(ISI)

        return ISI

    def CV(self):
        """
        Compute coefficient of variation
        """

        ISI = self.ISI()
        CV = np.std(ISI) / np.mean(ISI)

        return CV

    def fano(self):
        """
        Compute fano factor
        """

        spiketrain = self.spike_train()
        fano = np.var(spiketrain) / np.mean(spiketrain)

        return fano

    def ISI_shuffle(self, random_state=None):
        """
        Shuffle the ISI and return new spike times
        :param st: spike times
        :return:
        """

        # initialize empty array
        st = np.zeros_like(self.times)

        rng = np.random.RandomState(random_state)

        ISI = np.diff(self.times)

        st[1:] = np.cumsum(rng.permutation(ISI)) + st[0]

        return st

    @staticmethod
    def poisson_train(frequency, duration, start_time=0, seed=None):
        """Generator function for a Homogeneous Poisson train. I pulled this from somewhere but cannot remember
        where :(

        :param frequency: The mean spiking frequency.
        :param duration: Maximum duration.
        :param start_time: Timestamp.
        :param seed: Seed for the random number generator. If None, this will be
                decided by np, which chooses the system time.

        :return: A relative spike time from t=start_time, in seconds (not ms).

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
