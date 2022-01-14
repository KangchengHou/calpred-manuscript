import numpy as np
import pandas as pd
import random
from scipy import stats


def stratify_calculate_r2(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    stratify_col: str,
    n_boostrap: int = 10,
) -> pd.DataFrame:
    """
    Stratify dataframe by `stratify_col` with levels; within each level of the
    stratification, calculate the R2 value of the regression of `y_col` on `x_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `x_col`, `y_col` and
        `stratify_col`.
    x_col : str
        Name of the column containing the predictor variable.
    y_col : str
        Name of the column containing the response variable.
    stratify_col : str
        Name of the column containing the stratification variable.
    n_boostrap : int
        Number of bootstrap samples to draw. Default is 10.

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and three columns `level`, `R2`, and `R2_std`
        - level: imputed levels based on stratify_col
        - R2: R2 between x_col and y_col within each level
        - R2_std: standard deviation of R2 within each level obtained with bootstrap

    References
    ----------
    For bootstrap standard deviation, see:
    https://stats.stackexchange.com/questions/172662/how-do-you-calculate-the-standard-error-of-r2
    """

    grp_index_li = []
    if len(np.unique(df[stratify_col])) > 5:
        n_level = 5
        level_li = [i * (1 / n_level) for i in range(n_level + 1)]
        qtl_li = df[stratify_col].quantile(level_li).values

        for grp_i in range(n_level):
            grp_index_li.append(
                df.index[
                    (qtl_li[grp_i] <= df[stratify_col])
                    & (df[stratify_col] <= qtl_li[grp_i + 1])
                ]
            )

        for grp_i in range(n_level):
            df.loc[grp_index_li[grp_i], "level"] = int(grp_i + 1)

    else:
        n_level = len(np.unique(df[stratify_col]))
        df["level"] = df[stratify_col]
        for val in np.unique(df[stratify_col]):
            grp_index_li.append(df.index[df.level == val])

    grp_r2_samples = []
    for _ in range(n_boostrap):
        grp_index_random = []
        grp_r2_li = []
        for i in range(n_level):
            grp_index_random.append(
                random.choices(grp_index_li[i], k=len(grp_index_li[i]))
            )
            res = stats.linregress(
                df.loc[grp_index_random[i], x_col], df.loc[grp_index_random[i], y_col]
            )
            grp_r2_li.append(res.rvalue ** 2)
        grp_r2_samples.append(grp_r2_li)

    avg_grp_r2 = np.mean(grp_r2_samples, axis=0)
    std_grp_r2 = np.std(grp_r2_samples, axis=0)
    d = {"level": range(1, n_level + 1), "R2": avg_grp_r2, "R2_std": std_grp_r2}
    output = pd.DataFrame(data=d)
    return output


def eval_calibration(
    df: pd.DataFrame,
    x_col: str,
    lower_col: str,
    upper_col: str,
    stratify_col: str,
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
    lower_col : str
        Name of the column containing the prediction lower quantile variable.
    upper_col : str
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
        level_li = [i * (1 / n_level) for i in range(n_level + 1)]
        qtl_li = df[stratify_col].quantile(level_li).values

        for grp_i in range(n_level):
            grp_index_li.append(
                df.index[
                    (qtl_li[grp_i] <= df[stratify_col])
                    & (df[stratify_col] <= qtl_li[grp_i + 1])
                ]
            )

        for grp_i in range(n_level):
            df.loc[grp_index_li[grp_i], "level"] = int(grp_i + 1)

    else:
        n_level = len(np.unique(df[stratify_col]))
        df["level"] = df[stratify_col]
        for val in np.unique(df[stratify_col]):
            grp_index_li.append(df.index[df.level == val])

    grp_hits_li = []
    for i in range(n_level):
        grp_hits_li.append(
            (df[lower_col].values < df[x_col].values)
            & (df[x_col].values < df[upper_col].values)
        )

    grp_hits_prop_li = []
    for i in range(n_level):
        grp_hits_prop_li.append(np.mean(grp_hits_li[i]))

    d = {"level": range(1, n_level + 1), "coverage": grp_hits_prop_li}
    output = pd.DataFrame(data=d)
    return output
