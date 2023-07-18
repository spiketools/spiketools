"""Settings for tests."""

import os
from pathlib import Path

###################################################################################################
###################################################################################################

# Define some default values
FS = 100
N_SAMPLES = 100

# Set paths for test files
TESTS_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_TEST_FILE_PATH = TESTS_PATH / 'test_files'
TEST_PLOTS_PATH = BASE_TEST_FILE_PATH / 'plots'
