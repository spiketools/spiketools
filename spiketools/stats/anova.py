"""ANOVA related helper functions."""

import numpy as np
import pandas as pd

from spiketools.modutils.dependencies import safe_import, check_dependency

sm = safe_import('.api', 'statsmodels')
smf = safe_import('.formula.api', 'statsmodels')

###################################################################################################
###################################################################################################

def create_dataframe(data, columns, drop_na=True):
    """Create a dataframe from an array of data.

    Parameters
    ----------
    data : 2d array
        Array of data to organize into a dataframe.
    columns : list of str
        The column labels for the dataframe.
    drop_na : bool, optional, default: True
        Whether to drop NaN values from the dataframe.

    Returns
    -------
    df : pd.DataFrame
        Constructed dataframe.
    """

    n_trials, n_bins = data.shape
    labels = np.tile(np.array(range(0, n_bins)), n_trials)
    df = pd.DataFrame(np.transpose(np.vstack([labels, data.flatten()])),
                      columns=columns)

    if drop_na:
        df = df.dropna()

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
    f_val : float
        The F-value statistic of the ANOVA model.
        Returned if `return_type` is 'f_val'.
    results : pd.DataFrame
        The results of the model fit.
        Returned if `return_type` is 'results'.
    model : statsmodels object
        The fit model object.
        Returned if `return_type` is 'model'.
    """

    model = smf.ols(formula, data=df).fit()

    if return_type == 'model':
        return model

    results = sm.stats.anova_lm(model, typ=anova_type)

    if return_type == 'results':
        return results

    if return_type == 'f_val':
        return results['F'][feature]
