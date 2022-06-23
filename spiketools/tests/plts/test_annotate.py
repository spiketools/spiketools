"""Tests for spiketools.plts.annotate"""

import numpy as np
import matplotlib.pyplot as plt

from spiketools.tests.tutils import plot_test

from spiketools.plts.annotate import *
from spiketools.plts.annotate import (_add_vlines, _add_vshade, _add_hshade, _add_box_shade,
                                      _add_box_shades, _add_dots, _add_significance)

###################################################################################################
###################################################################################################

def test_color_pval():

    out1 = color_pval(0.025)
    assert out1 == 'red'

    out2 = color_pval(0.50)
    assert out2 == 'black'

    out3 = color_pval(0.005, 0.01, 'green')
    assert out3 == 'green'

@plot_test
def test_add_vlines():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    _add_vlines([1.5, 2.5, 3.5], ax=ax)

@plot_test
def test_add_vshade():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    _add_vshade([2., 3.], ax=ax)

@plot_test
def test_add_hshade():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    _add_hshade([2., 3.], ax=ax)


@plot_test
def test_add_box_shade():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    _add_box_shade(1.5, 2.5, 2, ax=ax)

@plot_test
def test_add_box_shades():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    _add_box_shades(np.array([1.5, 2.5]), np.array([1.5, 2.5]), ax=ax)

@plot_test
def test_add_dots():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    _add_dots(np.array([[1, 2], [2, 3]]), ax=ax)

@plot_test
def test_add_significance():

    _, ax = plt.subplots()

    x_values = [1, 2, 3, 4]
    stats = [1., 0.01, 0.5, 1.]

    ax.plot(x_values)
    _add_significance(stats, ax=ax)
