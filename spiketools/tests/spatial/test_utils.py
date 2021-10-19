"""Tests for spiketools.spatial.utils"""

import numpy as np

from spiketools.spatial.utils import *

###################################################################################################
###################################################################################################

def test_get_pos_ranges():

    positions = np.array([[1, 2, 3, 4, 5],
                          [5, 6, 7, 8, 9]])

    ranges = get_pos_ranges(positions)
    ranges[0] == [1, 5]
    ranges[1] == [5, 9]

def test_get_bin_width():

    bins = [1., 2., 3., 4., 5.]
    binw = get_bin_width(bins)
    assert binw == 1.
