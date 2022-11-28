"""Tests for spiketools.utils.run"""

from spiketools.utils.run import *

###################################################################################################
###################################################################################################

def test_create_methods_list():

    methods = {'place' : ['info', 'anova'], 'target' : ['info']}
    methods_lst = create_methods_list(methods)
    assert isinstance(methods_lst, list)
    assert methods_lst == ['place_info', 'place_anova', 'target_info']
