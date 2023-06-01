"""Tests for spiketools.spatial.speed"""

import numpy as np

from spiketools.spatial.speed import *

###################################################################################################
###################################################################################################

def test_compute_speed():

    timestamps1 = np.array([0., 1., 2., 3., 4.])
    timestamps2 = np.array([0., 1., 2., 2.5, 3.0])

    # 1d
    pos1d = np.array([0, 0, 1, 1, 2])

    out1d1 = compute_speed(pos1d, timestamps1, align_output=False)
    assert isinstance(out1d1, np.ndarray)
    assert len(out1d1) == pos1d.shape[-1] - 1
    assert np.allclose(out1d1, np.array([0, 1, 0, 1]))
    # Test with alignment
    assert np.allclose(compute_speed(pos1d, timestamps1), np.array([0, 0, 1, 0, 1]))


    out1d2 = compute_speed(pos1d, timestamps2, align_output=False)
    assert isinstance(out1d2, np.ndarray)
    assert len(out1d2) == pos1d.shape[-1] - 1
    assert np.allclose(out1d2, np.array([0, 1, 0, 2]))

    # 2d
    pos2d = np.array([[0, 0, 1, 1, 2],
                      [0, 0, 0, 1, 2]])

    out2d1 = compute_speed(pos2d, timestamps1, align_output=False)
    assert isinstance(out2d1, np.ndarray)
    assert len(out2d1) == pos2d.shape[-1] - 1
    assert np.allclose(out2d1, np.array([0, 1, 1, np.sqrt(2)]))
    # Test with alignment
    assert np.allclose(compute_speed(pos2d, timestamps1),
                       np.array([0, 0, 1, 1, np.sqrt(2)]))

    out2d2 = compute_speed(pos2d, timestamps2, align_output=False)
    assert isinstance(out2d2, np.ndarray)
    assert len(out2d2) == pos2d.shape[-1] - 1
    assert np.allclose(out2d2, np.array([0, 1, 1 / 0.5, np.sqrt(2) / 0.5]))
