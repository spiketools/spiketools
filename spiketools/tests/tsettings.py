"""Settings for tests."""

import os
import pkg_resources as pkg

###################################################################################################
###################################################################################################

# Define some default values
FS = 100
N_SAMPLES = 100

# Set paths for test files
BASE_TEST_FILE_PATH = pkg.resource_filename(__name__, 'test_files')
TEST_PLOTS_PATH = os.path.join(BASE_TEST_FILE_PATH, 'plots')
