"""Tests for spike_tools.objects.session"""

from spiketools.objects.session import *

###################################################################################################
###################################################################################################

def test_session():

    assert Session('SubjectCode', 'SessionCode', 'TaskCode')
