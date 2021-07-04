"""Tests for spiketools.objects.cell"""

from spiketools.objects.cell import *

###################################################################################################
###################################################################################################

def test_cell(tspikes):

    assert Cell(subject='SubjectCode',
                session='SessionCode',
                task='TaskCode',
                channel='ChannelCode',
                region='RegionCode',
                spikes=tspikes,
                cluster=None)

def test_cell_spike_train(tcell):

    spike_train = tcell.spike_train()

def test_cell_firing_rate(tcell):

    rate = tcell.firing_rate()

def test_cell_ISI(tcell):

    isis = tcell.ISI()

def test_cell_CV(tcell):

    cv = tcell.CV()

def test_cell_fano(tcell):

    fano = tcell.fano()

def test_cell_shuffle(tcell):

    shuffle = tcell.shuffle()
