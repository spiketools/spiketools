"""Pytest configuration file for testing spiketools."""

import pytest

import numpy as np

from spiketools.objects import Cell
from spiketools.utils import set_random_seed

###################################################################################################
###################################################################################################

def pytest_configure(config):

    set_random_seed(42)


@pytest.fixture(scope='session')
def tspikes():

    yield np.array([0.0, 0.5, 1.5, 2., 2.5, 3., 4., 5.]) * 1000

@pytest.fixture(scope='session')
def tcell(tspikes):

    yield Cell(subject='SubjectCode',
               session='SessionCode',
               task='TaskCode',
               channel='ChannelCode',
               region='RegionCode',
               spikes=tspikes,
               cluster=None)

@pytest.fixture(scope='session')
def tsession():

    yield Session(subject='SubjectCode',
                  session='SessionCode',
                  task='TaskCode')
