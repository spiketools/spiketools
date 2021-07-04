"""Pytest configuration file for testing spiketools."""

import os
import shutil
import pytest

import numpy as np

from spiketools.objects import Cell
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
