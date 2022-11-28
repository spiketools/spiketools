"""Tests for spiketools.objects.session"""

from spiketools.objects.session import *

###################################################################################################
###################################################################################################

def test_session(tunit):

    assert Session(subject='SubjectCode',
                   session='SessionCode',
                   task='TaskCode',
                   units=[tunit, tunit])

def test_session_add_unit(tsession, tunit):

    tsession.add_unit(tunit)
    assert tsession.units
    assert len(tsession.units) == 1 == tsession.n_units
    tsession.add_unit(tunit)
    assert tsession.units
    assert len(tsession.units) == 2 == tsession.n_units
