"""Tests for spiketools.plts.annotate"""

import matplotlib.pyplot as plt

from spiketools.tests.tutils import plot_test

from spiketools.plts.annotate import _add_significance_to_plot

###################################################################################################
###################################################################################################

@plot_test
def test_add_significance_to_plot():

    _, ax = plt.subplots()

    x_values = [1, 2, 3, 4]
    stats = [1., 0.01, 0.5, 1.]

    ax.plot(x_values)
    _add_significance_to_plot(stats, ax=ax)
