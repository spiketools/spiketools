"""Tests for spiketools.utils.base"""

from spiketools.utils.base import *

###################################################################################################
###################################################################################################

def test_flatten():

    lsts = [[1, 2], [3, 4]]
    assert flatten(lsts) == [1, 2, 3, 4]

def test_lower_list():

    lst = ['A', 'b', 'C', 'd']
    out = lower_list(lst)
    assert out == ['a', 'b', 'c', 'd']

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

def test_combine_dicts():

    d1 = {'a' : 1, 'b' : 2}
    d2 = {'c' : 3, 'd' : 4}
    d3 = {'e' : 5, 'f' : 6}

    out1 = combine_dicts([d1, d2])
    for cdict in [d1, d2]:
        for key, value in cdict.items():
            assert out1[key] == value

    out2 = combine_dicts([d1, d2, d3])
    for cdict in [d1, d2, d3]:
        for key, value in cdict.items():
            assert out2[key] == value

def test_add_key_prefix():

    tdict = {'a' : 1, 'b' : 2}
    prefix = 'test'
    out = add_key_prefix(tdict, prefix)

    for key, value in tdict.items():
        assert out[prefix + '_' + key] == value
