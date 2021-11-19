"""Tests for spiketools.plts.annotate"""

import matplotlib.pyplot as plt

from spiketools.tests.tutils import plot_test

from spiketools.plts.annotate import _add_significance_to_plot, _add_vlines

###################################################################################################
###################################################################################################

@plot_test
def test_add_significance_to_plot():

    _, ax = plt.subplots()

    x_values = [1, 2, 3, 4]
    stats = [1., 0.01, 0.5, 1.]

    ax.plot(x_values)
    _add_significance_to_plot(stats, ax=ax)

@plot_test
def test_add_vlines():

    _, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
    _add_vlines([1.5, 2.5, 3.5], ax=ax)
