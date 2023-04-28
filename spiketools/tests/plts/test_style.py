"""Tests for spiketools.plts.style."""

import numpy as np

from spiketools.plts.settings import SET_KWARGS

from spiketools.plts.style import *

###################################################################################################
###################################################################################################

def test_set_plt_kwargs():

    @set_plt_kwargs
    def example_plot():
        plt.plot([1, 2], [3, 4])

    xlim = (0, 2)
    ylim = (0, 4)
    title = 'Test Title'

    example_plot(xlim=xlim, ylim=ylim)
    assert plt.gca().get_xlim() == xlim
    assert plt.gca().get_ylim() == ylim

    example_plot(title=title)
    assert plt.gca().get_title() == title


def test_drop_spines():

    _, ax = plt.subplots()
    drop_spines(['top', 'right'], ax)

def test_invert_axes():

    # test inverting x & y axes separately
    _, ax1 = plt.subplots()
    ax1.plot([1, 2], [3, 4])

    invert_axes('x', ax1)
    assert ax1.get_xlim()[0] > ax1.get_xlim()[1]

    invert_axes('y', ax1)
    assert ax1.get_ylim()[0] > ax1.get_ylim()[1]

    # test inverting both axes together
    _, ax2 = plt.subplots()
    ax2.plot([1, 2], [3, 4])
    invert_axes('both', ax2)
    assert ax2.get_xlim()[0] > ax2.get_xlim()[1]
    assert ax2.get_ylim()[0] > ax2.get_ylim()[1]
