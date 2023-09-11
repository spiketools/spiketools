"""Tests for spiketools.utils.base"""

import numpy as np

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

def test_drop_key_prefix():

    tdict = {'test_a' : 1, 'test_b' : 2}
    prefix = 'test'
    out = drop_key_prefix(tdict, prefix)

    for key, value in out.items():
        assert prefix not in key
        assert tdict[prefix + '_' + key] == out[key]

def test_relabel_keys():

    indict = {'a' : 1, 'b' : 2}
    new_keys = {'a' : 'c'}

    outdict = relabel_keys(indict, new_keys)
    assert isinstance(outdict, dict)
    assert len(outdict) == len(indict)
    for key, val in new_keys.items():
        assert key not in outdict
        assert val in outdict

def test_subset_dict():

    indict = {'a_1' : 1, 'a_2' : 2, 'b_1' : 3, 'b_2' : 4}
    label = 'a'

    out = subset_dict(indict, label)
    assert out == {'a_1' : 1, 'a_2' : 2}
    assert indict == {'b_1' : 3, 'b_2' : 4}

def test_check_keys():

    indict = {'a_1' : 1, 'a_2' : 2, 'b_1' : 3, 'b_2' : 4}

    keys1 = ['a_0', 'a_1']
    out1 = check_keys(indict, keys1)
    assert out1 == 'a_1'

    keys2 = ['c_0', 'c_1']
    out2 = check_keys(indict, keys2)
    assert out2 is None

def test_listify():

    assert listify('test') == ['test']
    assert listify(['test']) == ['test']
    assert listify(0.5) == [0.5]
    assert listify([0.5]) == [0.5]

    # Test with array inputs
    assert listify(np.array([1, 2, 3, 4])) == [1, 2, 3, 4]
    out2d = listify(np.array([[1, 2], [3, 4]]))
    assert isinstance(out2d, list)
    assert len(out2d) == 1
    assert np.array_equal(out2d[0], np.array([[1, 2], [3, 4]]))

    # Test with indexing
    listify([1, 2], index=True) == [[1, 2]]
    listify([[1, 2]], index=True) == [[1, 2]]
