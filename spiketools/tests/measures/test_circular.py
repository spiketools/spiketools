"""Tests for spiketools.measures.circular"""

import numpy as np

from spiketools.measures.circular import *

###################################################################################################
###################################################################################################

def test_bin_circular():

    data = np.array([10, 50, 60, 80, 120, 150, 190, 200, 210, 225, 250, 275, 300, 325, 350])

    bin_edges, counts = bin_circular(data)

    assert len(bin_edges) - 1 == len(counts)
