"""Unit object."""

from spiketools.stats.shuffle import shuffle_spikes
from spiketools.measures.conversions import convert_times_to_train
from spiketools.measures.spikes import (compute_isis, compute_cv,
                                        compute_fano_factor, compute_firing_rate)

###################################################################################################
###################################################################################################

class Unit():
    """A Unit object.

    Parameters
    ----------
    uid : str
        Unit ID.
    spikes : 1d array
        Spike times, in seconds.
    channel : str
        Channel label.
    region : str
        Region label.
    cluster : 1d array
        Spike cluster.
    """

    def __init__(self, uid=None, spikes=None, channel=None, region=None, cluster=None):
        """Initialize a Unit object."""

        self.uid = uid
        self.channel = channel
        self.region = region
        self.spikes = spikes
        self.cluster = cluster


    def spike_train(self):
        """Convert spike times into a spike train vector (binary)."""

        return convert_times_to_train(self.spikes)


    def firing_rate(self):
        """Compute average firing rate."""

        return compute_firing_rate(self.spikes)


    def isis(self):
        """Compute inter-spike intervals."""

        return compute_isis(self.spikes)


    def cv(self):
        """Compute coefficient of variation."""

        return compute_cv(self.isis())


    def fano(self):
        """Compute fano factor."""

        return compute_fano_factor(self.spike_train())


    def shuffle(self, approach='ISI', n_shuffles=1000, **kwargs):
        """Shuffle spikes to create surrogates.

        Parameters
        ----------
        approach : {'ISI', 'BINCIRC', 'POISSON', 'CIRCULAR'}
            Which approach to take for shuffling spike times.
        n_shuffles : int, optional, default: 1000
            The number of shuffles to create.
        kwargs
            Additional keyword arguments for the shuffle functions.
            See `shuffle_spikes` for details.
        """

        return shuffle_spikes(self.spikes, approach, n_shuffles, **kwargs)
