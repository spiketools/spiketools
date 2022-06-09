"""Tests for spiketools.objects.session"""

from spiketools.objects.session import *

###################################################################################################
###################################################################################################

def test_session(tcell):

    assert Session(subject='SubjectCode',
                   session='SessionCode',
                   task='TaskCode',
                   units=[tcell, tcell])

def test_session_add_unit(tsession, tcell):

    tsession.add_unit(tcell)
    assert tsession.units
    assert len(tsession.units) == 1 == tsession.n_units
    tsession.add_unit(tcell)
    assert tsession.units
    assert len(tsession.units) == 2 == tsession.n_units
