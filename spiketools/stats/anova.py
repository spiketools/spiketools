"""ANOVA related helper functions."""

import numpy as np
import pandas as pd

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
