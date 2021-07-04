"""Tests for spiketools.objects.session"""

from spiketools.objects.session import *

###################################################################################################
###################################################################################################

def test_session():

    assert Session(subject='SubjectCode',
                   session='SessionCode',
                   task='TaskCode')
