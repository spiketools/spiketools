"""ANOVA related helper functions."""

from copy import deepcopy

import numpy as np
import pandas as pd

from spiketools.utils.base import flatten
from spiketools.utils.checks import check_param_options
from spiketools.modutils.dependencies import safe_import, check_dependency

sm = safe_import('.api', 'statsmodels')
smf = safe_import('.formula.api', 'statsmodels')

###################################################################################################
###################################################################################################

def create_dataframe(data, columns=None, drop_na=True, types=None):
    """Create a dataframe from an array of data.

    Parameters
    ----------
    data : dict or 2d array
        Data to organize into a dataframe.
        If dict, each key, value pairing should be a label and data array.
        If array, then should be organized as [n_observations, n_features].
    columns : list of str, optional
        The column labels for the dataframe.
        To be used if `data` is an array.
    drop_na : bool, optional, default: True
        Whether to drop NaN values from the dataframe.
    types : dict, optional


    Returns
    -------
    df : pd.DataFrame
        Constructed dataframe.
    """

    df = pd.DataFrame(data, columns=columns)

    if drop_na:
        df = df.dropna()

    if types:
        for column, ntype in types.items():
            df[column] = df[column].astype(ntype)

    return df


def create_dataframe_bins(bin_data, columns=None, other_data=None, drop_na=True):
    """Create a dataframe from an array of binned data.

    Parameters
    ----------
    bin_data : 2d or 3d array
        An array of data organized into pre-computed bins.
        If a 2d array, should be organized as [n_trials, n_bins].
        If a 3d array, should be organized as [n_trials, n_xbins, n_ybins].
    columns : list of str, optional
        The column labels for the bin data.
        Defaults to ['bin', 'fr'] for 1d bins or ['xbin', 'ybin' 'fr'] for 2d bins.
    other_data : dict
        Additional data columns, reflecting data per trial, to add to the dataframe.
        Each key should be a column label and each value should be an array of length n_trials.
    drop_na : bool, optional, default: True
        Whether to drop NaN values from the dataframe.

    Returns
    -------
    df : pd.DataFrame
        Constructed dataframe.
    """

    if bin_data.ndim == 2:

        df_columns = ['bin', 'fr'] if not columns else deepcopy(columns)

        n_trials, n_bins = bin_data.shape

        trial = np.repeat(np.arange(0, n_trials), n_bins)
        labels = np.tile(np.arange(0, n_bins), n_trials)

        df_data = np.stack([trial, labels, bin_data.flatten()], axis=1)

    elif bin_data.ndim == 3:

        df_columns = ['xbin', 'ybin', 'fr'] if not columns else deepcopy(columns)

        n_trials, n_xbins, n_ybins = bin_data.shape
        n_bins = n_xbins * n_ybins

        trial = np.repeat(np.arange(0, n_trials), n_xbins * n_ybins)
        xlabels = np.tile(np.repeat(np.arange(0, n_xbins), n_ybins), n_trials)
        ylabels = np.tile(np.arange(0, n_ybins), n_trials * n_xbins)

        df_data = np.stack([trial, xlabels, ylabels, bin_data.flatten()], axis=1)

    if other_data is not None:
        other_data = {label : np.repeat(data, n_bins) for label, data in other_data.items()}
        df_columns.extend(other_data.keys())
        df_data = np.hstack([df_data, np.array(list(other_data.values())).T])

    df_columns.insert(0, 'trial')
    types = {column : 'int' for column in df_columns if 'trial' in column or 'bin' in column}
    df = create_dataframe(df_data, df_columns, drop_na=drop_na, types=types)

    # Reorder dataframe so that `trial` column is first and `fr` is at the end & sorted in between
    df = df[flatten([['trial'], sorted(list(set(df.columns) - set(['trial', 'fr']))), ['fr']])]

    return df


@check_dependency(sm, 'statsmodels')
def fit_anova(df, formula, feature=None, return_type='f_val', anova_type=2):
    """Fit an ANOVA.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of data to fit the ANOVA to.
    formula : str
        The formula
    feature : str, optional
        Which feature to extract from the model.
        Only used (and required) if `return_type` is 'f_val'.
    return_type : {'f_val', 'results', 'model'}
        What to return after the model fitting. Options:

            * 'f_val' : returns the F-value for the requested feature
            * 'results' : returns the full set of model results
            * 'model' : returns the fit model object
    anova_type : {2, 3, 1}
        Which type of ANOVA test to perform.
        See `statsmodels.stats.anova.anova_lm` for details.

    Returns
    -------
    output : float or pd.DataFrame or statsmodels object
        If `return_type` is 'f_val', the f-value statistic of the ANOVA model.
        If `return_type` is 'results', the results of the model fit.
        If `return_type` is 'model', the fit model object.
    """

    check_param_options(return_type, 'return_type', ['model', 'results', 'f_val'])

    model = smf.ols(formula, data=df).fit()

    if return_type == 'model':
        output = model

    if return_type in ['results', 'f_val']:

        results = sm.stats.anova_lm(model, typ=anova_type)

        if return_type == 'results':
            output = results

        if return_type == 'f_val':
            output = results['F'][feature]

    return output
