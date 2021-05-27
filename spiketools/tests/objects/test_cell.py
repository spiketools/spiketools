"""Tests for spike_tools.objects.cell"""

from spiketools.objects.cell import *

###################################################################################################
###################################################################################################

def test_cell():

    assert Cell('SubjectCode', 'SessionCode', 'TaskCode', 'ChannelCode', 'RegionCode')
