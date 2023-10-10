"""Pytest configuration file for testing spiketools."""

import os
import shutil
import pytest

import numpy as np

from spiketools.objects import Unit, Session
from spiketools.utils import set_random_seed
from spiketools.tests.tsettings import BASE_TEST_FILE_PATH, TEST_PLOTS_PATH

###################################################################################################
###################################################################################################

## TEST SETUP

def pytest_configure(config):

    set_random_seed(42)

@pytest.fixture(scope='session', autouse=True)
def check_dir():
    """Once, prior to session, this will clear and re-initialize the test file directories."""

    # If the directories already exist, clear them
    if os.path.exists(BASE_TEST_FILE_PATH):
        shutil.rmtree(BASE_TEST_FILE_PATH)

    # Remake (empty) directories
    os.mkdir(BASE_TEST_FILE_PATH)
    os.mkdir(TEST_PLOTS_PATH)

## TEST OBJECTS

@pytest.fixture(scope='session')
def tspikes():

    yield np.array([0.5, 1.5, 2.0, 2.5, 3.0, 3.2, 3.7, 4.0, 4.2, 4.7,
                    5.0, 5.7, 6.0, 7.0, 7.5, 8.0, 8.2, 8.7, 9.2, 9.9])

@pytest.fixture(scope='session')
def tspikes_offset():

    tspikes = np.array([0.5, 1.5, 2.0, 2.5, 3.0, 3.2, 3.7, 4.0, 4.2, 4.7,
                        5.0, 5.7, 6.0, 7.0, 7.5, 8.0, 8.2, 8.7, 9.2, 9.9])

    yield tspikes - 5

@pytest.fixture(scope='session')
def ttrial_spikes():

    yield np.array([-0.75, -0.5, -0.25, 0.25, 0.5, 0.55, 0.6, 0.75, 0.80, 0.90])

@pytest.fixture(scope='session')
def twaveform():

    yield np.array([0, 0, 0, 1, 2, 3, 4, 5, 3, 1, 0, 0])

@pytest.fixture(scope='session')
def tisis():

    yield np.array([0.1, 0.25, 0.4, 0.1, 0.05, 0.2, 0.125])

@pytest.fixture(scope='session')
def tdata():

    yield np.random.random(10)

@pytest.fixture(scope='session')
def tdata2d():

    yield np.random.random((5, 2))

@pytest.fixture(scope='session')
def tdata3d():

    yield np.random.random((2, 3, 4))

@pytest.fixture(scope='session')
def tunit(tspikes):

    yield Unit(uid='UnitID',
               spikes=tspikes,
               channel='ChannelCode',
               region='RegionCode',
               cluster=None)

@pytest.fixture(scope='session')
def tsession():

    yield Session(subject='SubjectCode',
                  session='SessionCode',
                  task='TaskCode')
