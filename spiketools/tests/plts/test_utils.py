"""Tests for spiketools.plts.utils."""

import os

from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.utils import *

###################################################################################################
###################################################################################################

def test_check_ax():

    # Check running with None Input
    ax = check_ax(None)

    # Check running with pre-created axis
    _, ax = plt.subplots()
    nax = check_ax(ax)
    assert nax == ax

    # Check creating figure of a particular size
    figsize = [5, 5]
    ax = check_ax(None, figsize=figsize)
    fig = plt.gcf()
    assert list(fig.get_size_inches()) == figsize

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

def test_make_axes():

    n_axes = 5

    axes = make_axes(n_axes)
    assert len(axes) == n_axes

    axes = make_axes(n_axes, 2, row_size=2, col_size=2, wspace=0.1, hspace=0.1)
    assert len(axes) == n_axes + 1
