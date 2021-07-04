"""Settings for tests."""

import os
import pkg_resources as pkg

###################################################################################################
###################################################################################################

# Set paths for test files
BASE_TEST_FILE_PATH = pkg.resource_filename(__name__, 'test_files')
TEST_PLOTS_PATH = os.path.join(BASE_TEST_FILE_PATH, 'plots')
