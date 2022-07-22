"""ANOVA related helper functions."""

import numpy as np
import pandas as pd

from spiketools.utils.base import flatten
from spiketools.utils.checks import check_param_options
from spiketools.modutils.dependencies import safe_import, check_dependency

sm = safe_import('.api', 'statsmodels')
smf = safe_import('.formula.api', 'statsmodels')

###################################################################################################
###################################################################################################

def create_dataframe(data, columns=None, dropna=True, dtypes=None):
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
    dropna : bool, optional, default: True
        Whether to drop NaN values from the dataframe.
    dtypes : dict, optional
        Data types to typecast columns to.
        Each key should be a column label, and each associated value the type to typecast to.

    Returns
    -------
    df : pd.DataFrame
        Constructed dataframe.
    """

    df = pd.DataFrame(data, columns=columns)

    if dropna:
        df = df.dropna()

    if dtypes:
        for column, dtype in dtypes.items():
            df[column] = df[column].astype(dtype)

    return df


def create_dataframe_bins(bin_data, other_data=None, dropna=True, dtypes=None, bin_columns=None):
    """Create a dataframe from an array of binned data.

    Parameters
    ----------
    bin_data : 2d or 3d array
        An array of data organized into pre-computed bins.
        If a 2d array, should be organized as [n_trials, n_bins].
        If a 3d array, should be organized as [n_trials, n_xbins, n_ybins].
    other_data : dict, optional
        Additional data columns, reflecting data per trial, to add to the dataframe.
        Each key should be a column label and each value should be an array of length n_trials.
    drop_na : bool, optional, default: True
        Whether to drop NaN values from the dataframe.
    dtypes : dict, optional
        Data types to typecast columns to.
        Each key should be a column label, and each associated value the type to typecast to.
    bin_columns : list of str, optional
        Custom column labels for the bin data.
        If not provided, defaults to ['bin', 'fr'] for 1d or ['xbin', 'ybin' 'fr'] for 2d bins.

    Returns
    -------
    df : pd.DataFrame
        Constructed dataframe.
    """

    if bin_data.ndim == 2:

        n_trials, n_bins = bin_data.shape

        trial = np.repeat(np.arange(0, n_trials), n_bins)
        labels = np.tile(np.arange(0, n_bins), n_trials)

        df_data = {'trial' : trial,
                   bin_columns[0] if bin_columns[0] else 'bin' : labels,
                   bin_columns[1] if bin_columns[1] else 'fr' : bin_data.flatten()}

    elif bin_data.ndim == 3:

        n_trials, n_xbins, n_ybins = bin_data.shape
        n_bins = n_xbins * n_ybins

        trial = np.repeat(np.arange(0, n_trials), n_xbins * n_ybins)
        xlabels = np.tile(np.repeat(np.arange(0, n_xbins), n_ybins), n_trials)
        ylabels = np.tile(np.arange(0, n_ybins), n_trials * n_xbins)

        df_data = {'trial' : trial,
                   bin_columns[0] if bin_columns[0] else 'xbin' : xlabels,
                   bin_columns[1] if bin_columns[1] else 'ybin' : ylabels,
                   bin_columns[2] if bin_columns[2] else 'fr' : bin_data.flatten()}

    if other_data is not None:
        for label, data in other_data.items():
            df_data[label] = np.repeat(data, n_bins)

    dtype_defaults = {col : 'int' for col in df_data.keys() if col == 'trial' or 'bin' in col}
    dtypes = {**dtypes, **dtype_defaults} if dtypes is not None else dtype_defaults

    df = create_dataframe(df_data, dropna=dropna, dtypes=dtypes)

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
        The formula.
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
