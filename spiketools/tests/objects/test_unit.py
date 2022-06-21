"""Tests for spiketools.objects.unit"""

from spiketools.objects.unit import *

###################################################################################################
###################################################################################################

def test_unit(tspikes):

    assert Unit(uid='UID',
                spikes=tspikes,
                channel='ChannelCode',
                region='RegionCode',
                cluster=None)

def test_unit_spike_train(tunit):

    spike_train = tunit.spike_train()

def test_unit_firing_rate(tunit):

    rate = tunit.firing_rate()

def test_unit_isis(tunit):

    isis = tunit.isis()

def test_unit_cv(tunit):

    cv = tunit.cv()

def test_unit_fano(tunit):

    fano = tunit.fano()

def test_unit_shuffle(tunit):

    shuffle = tunit.shuffle()
