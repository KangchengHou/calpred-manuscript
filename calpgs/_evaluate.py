import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def compute_group_stats(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    predstd_col: str = None,
    ci=0.90,
    group_col: str = None,
    n_bootstrap: int = 0,
    cor: str = "pearson",
    return_r2_diff=False,
):
    """
    Summarize the results of prediction:
    with R2, coverage, interval length for different groups.

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
    cor : str
        correlation method, default pearson, can be pearson, spearman
    return_r2_diff : bool
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

    if cor == "pearson":
        cor_func = stats.pearsonr
    elif cor == "spearman":
        cor_func = stats.spearmanr
    else:
        raise ValueError(f"cor must be pearson or spearman, got {cor}")
    ci_z = stats.norm.ppf((1 + ci) / 2)
    if group_col is not None:
        df_grouped = df.groupby(group_col)
        r2 = df_grouped.apply(lambda df: cor_func(df[pred_col], df[y_col])[0] ** 2)

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
        r2 = cor_func(df[pred_col], df[y_col])[0] ** 2
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
                compute_group_stats(
                    df.sample(frac=1, replace=True),
                    y_col,
                    pred_col,
                    predstd_col,
                    ci,
                    group_col,
                    n_bootstrap=0,
                )
            )
        if isinstance(df_res, pd.Series):
            df_res_se = pd.Series(
                np.dstack(bootstrap_dfs).std(axis=2).flatten(), index=df_res.index
            )
        else:
            df_res_se = pd.DataFrame(
                np.dstack(bootstrap_dfs).std(axis=2),
                index=df_res.index,  # type: ignore
                columns=df_res.columns,  # type: ignore
            )
        if (group_col is not None) and return_r2_diff:
            r2_diff = np.array(
                [d["r2"].iloc[-1] - d["r2"].iloc[0] for d in bootstrap_dfs]
            )
            return df_res, df_res_se, r2_diff
        else:
            return df_res, df_res_se
