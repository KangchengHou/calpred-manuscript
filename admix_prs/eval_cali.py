import numpy as np
import pandas as pd

def eval_calibration(
    df: pd.DataFrame, x_col: str, low: str = 'QUANTILE_10', upp: str = 'QUANTILE_90' anc: str, 
) -> pd.DataFrame:
    """
    Evaluate if our calibrated prediction interval covers the real data, across all ancestries.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `x_col` and `y_col`.
    x_col : str
        Name of the column containing the real data variable.
    low : str
        Name of the column containing the prediction lower quantile variable.
    upp : str
        Name of the column containing the prediction upper quantile variable.
    anc : str
        Name of the column containing the population ancestry variable.

    Returns
    -------
    pandas.DataFrame
        Dataframe with `n_anc` rows and one column `Coverage` 
        - Coverage: Coverage of our prediction interval for x_col in each ancestry population

    """


