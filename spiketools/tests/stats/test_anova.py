"""Tests for spiketools.stats.anova"""

import pandas as pd

from spiketools.stats.anova import *

###################################################################################################
###################################################################################################

def test_create_dataframe(tdata2d):

    df = create_dataframe(tdata2d, ['A', 'B'])
    assert isinstance(df, pd.DataFrame)


def test_fit_anova(tdata2d):

    df = create_dataframe(tdata2d, ['out', 'pred'])

    XX = fit_anova(df, 'out ~ pred', 'pred')
