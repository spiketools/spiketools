"""Tests for spiketools.stats.anova"""

import pandas as pd

from spiketools.stats.anova import *

###################################################################################################
###################################################################################################

def test_create_dataframe(tdata2d):

    # test array input
    labels = ['A', 'B']
    df = create_dataframe(tdata2d, labels)
    assert isinstance(df, pd.DataFrame)

    # test with type casting
    df = create_dataframe(tdata2d, labels, types={'B' : 'float32'})
    assert df['B'].dtype == 'float32'

    # test dictionary input
    data_dict = {'c1' : [1, 2, 3, 4], 'c2' : [1.5, 2.5, 3.5, 4.5], 'c3' : ['a', 'b', 'c', 'd']}
    df = create_dataframe(data_dict)
    assert isinstance(df, pd.DataFrame)

def test_create_dataframe_bins():

    # Create data - 3 trials, 2 bins, each bin value matches trial index
    data2d = np.array([[0, 0], [1, 1], [2, 2]])
    df = create_dataframe_bins(data2d, ['bin', 'fr'])
    assert isinstance(df, pd.DataFrame)
    for ind, row in enumerate(data2d):
        df[df['trial'] == ind]['fr'].values == row

    # Create new data, where each bin value matches bin index
    data2d = np.array([[0, 1, 2],
                       [0, 1, 2]])
    df = create_dataframe_bins(data2d, ['bin', 'fr'])
    for ind in range(0, np.max(data2d) + 1):
        assert np.all(df[df['bin'] == ind]['fr'].values == ind)

    # Check 3d array
    data3d = np.array([[[0, 1, 2], [3, 4, 5]],
                       [[6, 7, 8], [9, 10, 11]]])
    df = create_dataframe_bins(data3d, ['xbin', 'ybin', 'fr'])
    assert isinstance(df, pd.DataFrame)
    for trial, ax, bx, fr in zip(df['trial'], df['xbin'], df['ybin'], df['fr']):
        assert data3d[trial, ax, bx] == fr

def test_fit_anova(tdata2d):

    df = create_dataframe(tdata2d, ['out', 'pred'])

    f_val = fit_anova(df, 'out ~ pred', 'pred', return_type='f_val')
    assert isinstance(f_val, float)

    results = fit_anova(df, 'out ~ pred', 'pred', return_type='results')
    assert isinstance(results, pd.DataFrame)

    model = fit_anova(df, 'out ~ pred', 'pred', return_type='model')
    assert model
