"""Tests for spiketools.utils.base"""

from spiketools.utils.base import *

###################################################################################################
###################################################################################################

def test_flatten():

    lsts = [[1, 2], [3, 4]]
    assert flatten(lsts) == [1, 2, 3, 4]

def test_select_from_list():

    lst = [1, 2, 3, 4, 5]
    select = [True, False, True, False, True]
    out = select_from_list(lst, select)
    assert out == [1, 3, 5]
