"""Tests for spiketools.plts.task"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.task import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_task_structure():

    shades1 = [[2, 7], [4, 8]]
    shades2 = [[5.5, 8.5], [6.5, 9.5]]

    lines1 = [3, 6, 9]
    lines2 = [2.5, 5, 7.5]

    plot_task_structure(shades1, lines1,
                        file_path=TEST_PLOTS_PATH, file_name='tplot_task_structure1.png')

    plot_task_structure([shades1, shades2], [lines1, lines2],
                        shade_colors=['purple', 'orange'], line_colors=['red', 'black'],
                        file_path=TEST_PLOTS_PATH, file_name='tplot_task_structure2.png')


@plot_test
def test_plot_task_events():

    times1 = np.array([0.25, 0.75, 1.0, 1.25, 1.75])
    times2 = np.array([0.50, 1.25, 1.5, 2.0])

    plot_task_events(times1, color='red', alpha=0.5,
                     file_path=TEST_PLOTS_PATH, file_name='tplot_task_events1.png')

    plot_task_events([times1, times2],
                     file_path=TEST_PLOTS_PATH, file_name='tplot_task_events2.png')

    plot_task_events([{'times' : times1, 'color' : 'red', 'alpha' : 0.5},
                     {'times' : times2, 'color' : 'blue', 'alpha' : 0.5}],
                     file_path=TEST_PLOTS_PATH, file_name='tplot_task_events3.png')
