import numpy as np
import pandas as pd

def eval_calibration(
    df: pd.DataFrame, x_col: str, stratify_col: str, low: str = 'QUANTILE_10', upp: str = 'QUANTILE_90'
) -> pd.DataFrame:
    """
    Stratify dataframe by `stratify_col` with levels; within each level of the 
    stratification, evaluate if `x_col` calibrated prediction interval from `upp` 
    quantile to `low` quantile covers the real data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `x_col` and `y_col`.
    x_col : str
        Name of the column containing the real data variable.
    stratify_col : str
        Name of the column containing the stratification variable.
    low : str
        Name of the column containing the prediction lower quantile variable.
    upp : str
        Name of the column containing the prediction upper quantile variable.

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and two columns `level` and `Coverage` 
        - level: imputed levels based on stratify_col
        - Coverage: Coverage of our prediction interval for x_col in each ancestry population

    """

    grp_index_li = []
    if len(np.unique(df[stratify_col])) > 5:
        n_level = 5
        level_li = [i*(1/n_level) for i in range(n_level+1)]
        qtl_li = df[stratify_col].quantile(level_li).values

        for grp_i in range(n_level):
            grp_index_li.append(df.index[(qtl_li[grp_i] <= df[stratify_col]) & (df[stratify_col] <= qtl_li[grp_i+1])])

        for grp_i in range(n_level):
            df.loc[grp_index_li[grp_i], 'level'] = int(grp_i+1)

    else:
        n_level = len(np.unique(df[stratify_col]))
        df['level'] = df[stratify_col]
        for val in np.unique(df[stratify_col]):
            grp_index_li.append(df.index[df.level == val])

    grp_hits_li = []
    for i in range(n_level):
        grp_hits_li.append((df[low].values< df.bmi.values) & (df.bmi.values < df[upp].values))
    
    grp_hits_prop_li = []
    for i in range(n_level):
        grp_hits_prop_li.append(np.mean(grp_hits_li[i]))
    
    d = {'level': range(1, n_level+1), 'Coverage': grp_hits_prop_li}
    output = pd.DataFrame(data=d)
    return output
    