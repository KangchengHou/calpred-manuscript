import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.metrics import r2_score


def make_levels(
    df: pd.DataFrame,
    stratify_col: str,
    n_level: int,
) -> list:
    """
    Separate the dataframe by `stratify_col` into several levels;
    levels imputed by the number of distinctive values in `stratify_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `stratify_col`.
    stratify_col : str
        Name of the column to be stratified.
    n_level : int =

    Returns
    ----------
    list
        a list of indexes of levels.
    """

    level_li = [i * (1 / n_level) for i in range(n_level + 1)]
    cut_li = pd.qcut(df[stratify_col], level_li)

    return cut_li


def stratify_calculate_r2(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_boostrap: int = 10,
    group_col: str = None,
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
    group_col : str
        Name of the column containing the group variable.
    n_boostrap : int
        Number of bootstrap samples to draw. Default is 10.

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and three columns `group_col` / `indiv`, `R2`, and `R2_std`
        - `group_col` / `indiv`: the grp specified / all individuals without specification
        - R2: R2 between x_col and y_col within each level
        - R2_std: standard deviation of R2 within each level obtained with bootstrap
    """

    grp_index_li = []
    grp_r2_samples = []
    if group_col is not None:
        grp_li = []
        grp_li = np.unique(df[group_col])
        n_level = len(grp_li)
        for grp_i in range(n_level):
            grp_index_li.append(df.index[df[group_col] == grp_li[grp_i]])

        for _ in range(n_boostrap):
            grp_index_random = []
            grp_r2_li = []
            for i in range(n_level):
                grp_index_random.append(
                    random.choices(grp_index_li[i], k=len(grp_index_li[i]))
                )
                res = stats.linregress(
                    df.loc[grp_index_random[i], x_col],
                    df.loc[grp_index_random[i], y_col],
                )
                grp_r2_li.append(res.rvalue ** 2)
            grp_r2_samples.append(grp_r2_li)

    else:
        for _ in range(n_boostrap):
            grp_index_random = random.choices(df.index, k=int(len(df.index) * 0.8))
            res = stats.linregress(
                df.loc[grp_index_random, x_col], df.loc[grp_index_random, y_col]
            )
            grp_r2_samples.append(res.rvalue ** 2)

    avg_grp_r2 = np.mean(grp_r2_samples, axis=0)
    std_grp_r2 = np.std(grp_r2_samples, axis=0)

    if group_col is not None:
        d = {group_col: grp_li, "R2": avg_grp_r2, "R2_std": std_grp_r2}
        return pd.DataFrame(data=d)
    else:
        d = {"R2": [avg_grp_r2], "R2_std": [std_grp_r2]}
        return pd.Series(data=d)


def evaluate(
    df: pd.DataFrame,
    true_col: str,
    lower_col: str,
    upper_col: str,
    group_col: str = None,
):
    """
    Evaluate R2, coverage, interval length for different groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `true_col`, `lower_col`,
        `upper_col`, and `group_col`.
    true_col : str
        Name of the column containing the true value.
    lower_col : str
        Name of the column containing the lower bound of the interval.
    upper_col : str
        Name of the column containing the upper bound of the interval.
    group_col : str
        Name of the column containing the group variable.

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and three columns `group_col`, `r2`, `coverage`,
        and `interval_length`
    """
    if group_col is not None:
        df_grouped = df.groupby(group_col)
        r2 = df_grouped.apply(
            lambda df: stats.linregress(
                (df[lower_col] + df[upper_col]) / 2, df[true_col]
            ).rvalue
            ** 2
        )

        coverage = df_grouped.apply(
            lambda df: df[true_col].between(df[lower_col], df[upper_col]).mean()
        )
        length = df_grouped.apply(lambda df: (df[upper_col] - df[lower_col]).mean())
        df = {
            "r2": r2,
            "coverage": coverage,
            "length": length,
        }
        return pd.DataFrame(data=df)
    else:
        r2 = (
            stats.linregress((df[lower_col] + df[upper_col]) / 2, df[true_col]).rvalue
            ** 2
        )
        coverage = df[true_col].between(df[lower_col], df[upper_col]).mean()
        length = (df[upper_col] - df[lower_col]).mean()
        df = {
            "r2": r2,
            "coverage": coverage,
            "length": length,
        }
        return pd.Series(data=df)


def eval_calibration(
    df: pd.DataFrame,
    y_col: str,
    lower_col: str,
    upper_col: str,
    group_col: str = None,
) -> pd.DataFrame:
    """
    Stratify dataframe by `stratify_col` with levels; within each level of the
    stratification, evaluate if `x_col` calibrated prediction interval from `upp`
    quantile to `low` quantile covers the real data.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `x_col` and `y_col`.
    y_col : str
        Name of the column containing the real data variable.
    group_col : str
        Name of the column containing the group variable.
    lower_col : str
        Name of the column containing the prediction lower quantile variable.
    upper_col : str
        Name of the column containing the prediction upper quantile variable.

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and two columns `group_col` / `indiv` and `Coverage`
        - `group_col` / `indiv`: the grp specified / all individuals without specification
        - Coverage: Coverage of our prediction interval for x_col in each ancestry population

    """

    if group_col is not None:
        grp_index_li, grp_li = [], []
        grp_li = np.unique(df[group_col])
        n_level = len(grp_li)
        for grp_i in range(n_level):
            grp_index_li.append(df.index[df[group_col] == grp_li[grp_i]])

        grp_hits_li = []
        for i in range(n_level):
            grp_hits_li.append(
                (df.loc[grp_index_li[i], lower_col] < df.loc[grp_index_li[i], y_col])
                & (df.loc[grp_index_li[i], y_col] < df.loc[grp_index_li[i], upper_col])
            )

        grp_hits_prop_li = []
        for i in range(n_level):
            grp_hits_prop_li.append(np.mean(grp_hits_li[i]))

        d = {group_col: grp_li, "coverage": grp_hits_prop_li}
        return pd.DataFrame(data=d)
    else:
        res = np.array(
            (df.loc[:, lower_col] < df.loc[:, y_col])
            & (df.loc[:, y_col] < df.loc[:, upper_col])
        ).mean()
        d = {"coverage": res}
        return pd.Series(data=d)
