"""Tests for spiketools.plts.utils."""

import os

from spiketools.tests.tutils import fig_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.utils import *

###################################################################################################
###################################################################################################

def test_check_ax():

    # Check running with None Input
    ax = check_ax(None)
    assert ax is not None

    # Check running with pre-created axis
    _, ax = plt.subplots()
    nax = check_ax(ax)
    assert nax == ax

    # Check creating figure of a particular size
    figsize = [5, 5]
    ax = check_ax(None, figsize=figsize)
    fig = plt.gcf()
    assert list(fig.get_size_inches()) == figsize

    # Check getting current axis
    _, ax = plt.subplots()
    nax = check_ax(None, return_current=True)
    assert nax == ax

def test_savefig():

    @savefig
    def example_plot():
        plt.plot([1, 2], [3, 4])

    # Test defaults to saving given file path & name
    example_plot(file_path=TEST_PLOTS_PATH, file_name='test_savefig1.pdf')
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig1.pdf'))

    # Test works the same when explicitly given `save_fig`
    example_plot(save_fig=True, file_path=TEST_PLOTS_PATH, file_name='test_savefig2.pdf')
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig2.pdf'))

    # Test giving additional save kwargs
    example_plot(file_path=TEST_PLOTS_PATH, file_name='test_savefig3.pdf',
                 save_kwargs={'facecolor' : 'red'})
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig3.pdf'))

    # Test does not save when `save_fig` set to False
    example_plot(save_fig=False, file_path=TEST_PLOTS_PATH, file_name='test_savefig_nope.pdf')
    assert not os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig_nope.pdf'))

def test_save_figure():

    plt.plot([1, 2], [3, 4])
    save_figure(file_name='test_save_figure.pdf', file_path=TEST_PLOTS_PATH)
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_save_figure.pdf'))

@fig_test
def test_make_axes():

    n_axes = 5

    axes = make_axes(n_axes)
    assert len(axes) == n_axes

    axes = make_axes(n_axes, 2, row_size=2, col_size=2, wspace=0.1, hspace=0.1)
    assert len(axes) == n_axes + 1

def test_make_grid():

    nrows, ncols = 2, 2
    grid = make_grid(nrows, ncols, title='grid title')
    assert grid.nrows == nrows
    assert grid.ncols == ncols

@fig_test
def test_get_grid_subplot():

    grid = make_grid(2, 2)
    ax = get_grid_subplot(grid, 0, 0)
    assert ax

def test_invert_axes():

    # test inverting x & y axes separately
    _, ax1 = plt.subplots()
    ax1.plot([1, 2], [3, 4])

    invert_axes(ax1, 'x')
    assert ax1.get_xlim()[0] > ax1.get_xlim()[1]

    invert_axes(ax1, 'y')
    assert ax1.get_ylim()[0] > ax1.get_ylim()[1]

    # test inverting both axes together
    _, ax2 = plt.subplots()
    ax2.plot([1, 2], [3, 4])
    invert_axes(ax2, 'both')
    assert ax2.get_xlim()[0] > ax2.get_xlim()[1]
    assert ax2.get_ylim()[0] > ax2.get_ylim()[1]
