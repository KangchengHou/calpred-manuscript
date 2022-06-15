from pyexpat.errors import XML_ERROR_SYNTAX
from matplotlib.axis import XAxis
import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from tqdm import tqdm


def uct_metrics(pred_mean, pred_std, y):

    # Compute all uncertainty metrics
    metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y)
    return metrics


def summarize_pred(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    predstd_col: str = None,
    ci=0.90,
    group_col: str = None,
    n_bootstrap: int = 0,
    return_r2_diff=False,
):
    """
    Summarize the results of prediction, produce R2, coverage, interval length for
        different groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `y_col`, `pred_col`
    y_col : str
        Name of the column containing the true value.
    pred_col : str
        predicted value
    predstd_col : str
        standard deviation of the prediction
    group_col : str
        Name of the column containing the group variable.
    return_se : bool
        whether to return standard error of prediction,

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and three columns `group_col`, `r2`, `coverage`,
        and `interval_length`
    """
    val_cols = [y_col, pred_col]
    if group_col is not None:
        val_cols.append(group_col)
    if predstd_col is not None:
        val_cols.append(predstd_col)

    # dropping rows with missing values
    df = df[val_cols].dropna()

    ci_z = stats.norm.ppf((1 + ci) / 2)
    if group_col is not None:
        df_grouped = df.groupby(group_col)
        r2 = df_grouped.apply(
            lambda df: stats.linregress(df[pred_col], df[y_col]).rvalue ** 2
        )

        y_std = df_grouped.apply(lambda df: df[y_col].std())
        pred_std = df_grouped.apply(lambda df: df[pred_col].std())
        df_res = {
            "r2": r2,
            "std(y)": y_std,
            "std(pred)": pred_std,
        }

        if predstd_col is not None:
            df_res["coverage"] = df_grouped.apply(
                lambda df: df[y_col]
                .between(
                    df[pred_col] - df[predstd_col] * ci_z,
                    df[pred_col] + df[predstd_col] * ci_z,
                )
                .mean()
            )
            df_res["length"] = df_grouped.apply(
                lambda df: (df[predstd_col] * ci_z).mean()
            )
        df_res = pd.DataFrame(df_res)
    else:
        r2 = stats.linregress((df[pred_col]) / 2, df[y_col]).rvalue ** 2
        y_std = df[y_col].std()
        pred_std = df[pred_col].std()

        df_res = {
            "r2": r2,
            "std(y)": y_std,
            "std(pred)": pred_std,
        }
        if predstd_col is not None:

            df_res["coverage"] = (
                df[y_col]
                .between(
                    df[pred_col] - df[predstd_col] * ci_z,
                    df[pred_col] + df[predstd_col] * ci_z,
                )
                .mean()
            )
            df_res["length"] = (df[predstd_col] * ci_z).mean()
        df_res = pd.Series(df_res)
    if n_bootstrap == 0:
        return df_res
    else:
        bootstrap_dfs = []
        for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
            # sample with replacement
            bootstrap_dfs.append(
                summarize_pred(
                    df.sample(frac=1, replace=True),
                    y_col,
                    pred_col,
                    predstd_col,
                    ci,
                    group_col,
                    n_bootstrap=0,
                )
            )
        df_res_se = pd.DataFrame(
            np.dstack(bootstrap_dfs).std(axis=2),
            index=df_res.index,
            columns=df_res.columns,
        )
        if (group_col is not None) and return_r2_diff:
            r2_diff = np.array(
                [d["r2"].iloc[-1] - d["r2"].iloc[0] for d in bootstrap_dfs]
            )
            return df_res, df_res_se, r2_diff
        else:
            return df_res, df_res_se
