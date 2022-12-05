import calpgs
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import subprocess
from typing import List


def estimate_coef(
    df_path: str,
    y_col: str,
    mean_cols: List[str],
    var_cols: List[str],
    out_prefix: str,
    target_df_path: str = None,
):
    """Estimate coefficients using data from a file

    Parameters
    ----------
    df : str
        Path to a tab-separated file containing the data
    y_col : str
        Name of the column containing the outcome variable
    mean_cols : List[str]
        Names of the columns containing the mean values
    var_cols : List[str]
        Names of the columns containing the variance values
    out_prefix : str
        Prefix for the output files
    target_df : str, optional
        Path to a tab-separated file containing the data to be applied to perform calibration
        If None, the data from `df` will be used, by default None
    """
    df = pd.read_csv(df_path, sep="\t", index_col=0)
    if target_df_path is not None:
        target_df = pd.read_csv(target_df_path, sep="\t", index_col=0)
    else:
        target_df = df
    # format data set
    mean_covar = sm.add_constant(df[mean_cols])
    var_covar = sm.add_constant(df[var_cols])
    y = df[y_col].values
    fit = calpgs.fit_het_linear(
        y=y,
        mean_covar=mean_covar,
        var_covar=var_covar,
        slope_covar=None,
        return_est_covar=True,
        trace=True,
    )

    mean_coef, var_coef, mean_cov, var_cov = fit
    mean_se = np.sqrt(np.diag(mean_cov))
    var_se = np.sqrt(np.diag(var_cov))

    df_beta_params = pd.DataFrame(
        {"mean_coef": mean_coef, "mean_se": mean_se, "mean_z": mean_coef / mean_se},
        index=mean_covar.columns,
    )
    df_gamma_params = pd.DataFrame(
        {
            "var_coef": var_coef,
            "var_se": var_se,
            "var_z": var_coef / var_se,
        },
        index=var_covar.columns,
    )

    # estimated parameters
    df_params = pd.merge(
        df_beta_params, df_gamma_params, left_index=True, right_index=True, how="outer"
    )
    df_params.to_csv(out_prefix + ".params.tsv", sep="\t")

    # prediction
    pred_mean = sm.add_constant(target_df[mean_cols]).dot(mean_coef)
    pred_std = np.sqrt(np.exp(sm.add_constant(target_df[var_cols]).dot(var_coef)))
    assert ("pred_mean" not in target_df.columns) and (
        "pred_mean" not in target_df.columns
    )
    target_df["pred_mean"] = pred_mean
    target_df["pred_std"] = pred_std
    target_df.to_csv(out_prefix + ".pred.tsv", sep="\t")


def quantify_r2(
    df_path: str,
    y_col: str,
    pred_col: str,
    test_cols: List[str],
    out_prefix: str,
    predstd_col: str = None,
    n_bootstrap=None,
):
    """Quantify R2 using data from a file

    Parameters
    ----------
    df : str
        Path to a tab-separated file containing the data
    y_col : str
        Name of the column containing the outcome variable
    pred_col : str
        Name of the column containing the predicted values
    test_cols : List[str]
        Names of the columns containing the test values
    out_prefix : str
        Prefix for the output files
    """

    df = pd.read_csv(df_path, sep="\t", index_col=0)
    df_baseline = calpgs.compute_group_stats(
        df,
        y_col=y_col,
        pred_col=pred_col,
    )
    df_baseline.to_csv(out_prefix + ".baseline.tsv", sep="\t", header=False)

    tmp_file = out_prefix + ".tmp.tsv"
    df.to_csv(tmp_file, sep="\t")
    cmds = [
        "calpgs group-stats",
        f"--df {tmp_file}",
        f"--y {y_col}",
        f"--pred {pred_col}",
        f"--group {','.join(test_cols)}",
        "--cor spearman",
        f"--out {out_prefix}",
    ]
    if predstd_col is not None:
        cmds.append(f"--predstd {predstd_col}")
    if n_bootstrap is not None:
        cmds.append(f"--n-bootstrap {n_bootstrap}")
    subprocess.check_call(" ".join(cmds), shell=True)
    os.remove(tmp_file)