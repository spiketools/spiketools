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

def test_count_elements():

    lst = ['a', 'b', 'c', 'a']
    counter = count_elements(lst)
    assert set(counter.keys()) == set(lst)
    assert sum(counter.values()) == len(lst)

    lst = [0, 1, 3, 1, 4]
    counter = count_elements(lst, labels='count', sort=True)
    assert list(counter.keys()) == list(range(max(lst) + 1))
    assert sum(counter.values()) == len(lst)
