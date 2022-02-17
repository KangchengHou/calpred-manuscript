import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct



def uct_metrics(pred_mean, pred_std, y):

    # Compute all uncertainty metrics
    metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y)
    return metrics


def evaluate(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    predstd_col: str,
    ci=0.90,
    group_col: str = None,
):
    """
    Evaluate R2, coverage, interval length for different groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `true_col`, `lower_col`,
        `upper_col`, and `group_col`.
    y_col : str
        Name of the column containing the true value.
    pred_col : str
        predicted value
    std_col : str
        standard deviation of the prediction
    group_col : str
        Name of the column containing the group variable.

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and three columns `group_col`, `r2`, `coverage`,
        and `interval_length`
    """
    ci_z = stats.norm.ppf((1 + ci) / 2)
    if group_col is not None:
        df_grouped = df.groupby(group_col)
        r2 = df_grouped.apply(
            lambda df: stats.linregress(df[pred_col], df[y_col]).rvalue ** 2
        )

        coverage = df_grouped.apply(
            lambda df: df[y_col]
            .between(
                df[pred_col] - df[predstd_col] * ci_z,
                df[pred_col] + df[predstd_col] * ci_z,
            )
            .mean()
        )
        length = df_grouped.apply(lambda df: (df[predstd_col] * ci_z).mean())
        df = {
            "r2": r2,
            "coverage": coverage,
            "length": length,
        }
        return pd.DataFrame(data=df)
    else:
        r2 = stats.linregress((df[pred_col]) / 2, df[y_col]).rvalue ** 2
        coverage = (
            df[y_col]
            .between(
                df[pred_col] - df[predstd_col] * ci_z,
                df[pred_col] + df[predstd_col] * ci_z,
            )
            .mean()
        )
        length = (df[predstd_col] * ci_z).mean()
        df = {
            "r2": r2,
            "coverage": coverage,
            "length": length,
        }
        return pd.Series(data=df)
