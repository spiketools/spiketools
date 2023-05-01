"""Tests for spiketools.plts.annotate"""

import numpy as np
import matplotlib.pyplot as plt

from spiketools.tests.tutils import plot_test

from spiketools.plts.annotate import *

###################################################################################################
###################################################################################################

def test_color_pvalue():

    out1 = color_pvalue(0.025)
    assert out1 == 'red'

    out2 = color_pvalue(0.50)
    assert out2 == 'black'

    out3 = color_pvalue(0.005, 0.01, 'green')
    assert out3 == 'green'

@plot_test
def test_add_vlines():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    add_vlines([1.5, 2.5, 3.5], ax=ax)

@plot_test
def test_add_hlines():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    add_hlines([1.5, 2.5, 3.5], ax=ax)

@plot_test
def test_add_gridlines():

    _, ax1 = plt.subplots()
    ax1.plot([0, 2], [0, 2])

    bins = [0.5, 1.5]
    add_gridlines(bins, None, ax1)
    assert np.array_equal(ax1.get_xticks(), bins)
    assert np.array_equal(ax1.get_yticks(), [])
    add_gridlines(None, bins, ax1)
    assert np.array_equal(ax1.get_xticks(), [])
    assert np.array_equal(ax1.get_yticks(), bins)
    add_gridlines(bins, bins, ax1)
    assert np.array_equal(ax1.get_xticks(), bins)
    assert np.array_equal(ax1.get_yticks(), bins)

@plot_test
def test_add_vshades():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    add_vshades([2., 3.], ax=ax)
    add_vshades([[0.5, 0.75], [3.5, 3.75]], color='red', ax=ax)

@plot_test
def test_add_hshades():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    add_hshades([2., 3.], ax=ax)
    add_hshades([[0.5, 0.75], [3.5, 3.75]], color='red', ax=ax)

@plot_test
def test_add_box_shade():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    add_box_shade(1.5, 2.5, 2, ax=ax)

@plot_test
def test_add_box_shades():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    add_box_shades(np.array([1.5, 2.5]), np.array([1.5, 2.5]), ax=ax)

@plot_test
def test_add_dots():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    add_dots(np.array([[1, 2], [2, 3]]), ax=ax)

    # Test adding an additional dot from 1d array
    add_dots(np.array([2.5, 2.5]), color='red', ax=ax)

@plot_test
def test_add_significance():

    _, ax = plt.subplots()

    x_values = [1, 2, 3, 4]
    stats = [1., 0.01, 0.5, 1.]

    ax.plot(x_values)
    add_significance(stats, ax=ax)

@plot_test
def test_add_text_labels_x():

    _, ax = plt.subplots()
    data = [15, 12, 32, 22]
    ax.bar(['A', 'B', 'C', 'D'], data)
    add_text_labels(data, axis='x')

@plot_test
def test_add_text_labels_y():

    _, ax = plt.subplots()
    data = [15, 12, 32, 22]
    ax.barh(['A', 'B', 'C', 'D'], data)
    add_text_labels(data, axis='y', position='end')
