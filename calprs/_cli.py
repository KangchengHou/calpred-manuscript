import fire
from typing import Union, List
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger()


def log_params(name, params):
    logger.info(
        f"Received parameters: \n{name}\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in params.items())
    )


def r2diff(df: str, y: str, pred: str, group: str, out: str, n_bootstrap: int = 1000):
    """
    Calculate the difference between the r2 between `y` and `pred` across groups of
    individuals.

    Parameters
    ----------
    df : str
        Path to the dataframe containing the data.
    y : str
        Name of the column containing the observed phenotype.
    pred : str
        Name of the column containing the predicted value.
    group : Union[str, List[str]]
        Name of the column containing the group variable.
    out : str
        Path to the output file.
    n_bootstrap : int
        Number of bootstraps to perform, default 1000.
    """
    
    df = pd.read_csv(df, sep='\t', index_col=0)
    flag_li = isinstance(group, list)
    
    out_li = []
    if flag_li:
        for col in group:
            n_unique = len(np.unique(df[col].values))
            if n_unique > 5:
                df[col] = pd.qcut(df[col], q=5, duplicates="drop").cat.codes
                print(f"Converting column '{col}' to 5 quintiles")
            df_res, df_res_se, r2_diff = admix_prs.summarize_pred(
                df,
                y_col=y,
                pred_col=pred,
                group_col=col,
                n_bootstrap=n_bootstrap,
                return_r2_diff=True,
            )
            out_li.append(
                [col, df_res["r2"].iloc[-1] - df_res["r2"].iloc[0], np.mean(r2_diff > 0)]
            )
    else: 
        n_unique = len(np.unique(df[group].values))
        if n_unique > 5:
            df[group] = pd.qcut(df[group], q=5, duplicates="drop").cat.codes
            print(f"Converting column '{group}' to 5 quintiles")
        df_res, df_res_se, r2_diff = admix_prs.summarize_pred(
            df,
            y_col=y,
            pred_col=pred,
            group_col=group,
            n_bootstrap=n_bootstrap,
            return_r2_diff=True,
        )
        out_li.append(
            [group, df_res["r2"].iloc[-1] - df_res["r2"].iloc[0], np.mean(r2_diff > 0)]
        )

    df_out = pd.DataFrame(out_li, columns=["group", "r2diff", "prob>0"])     
    df_out.to_csv(out, sep='\t', index=False)


def model(
    df: str,
    y: str,
    pred: str,
    predstd: str,
    ci: float = 0.9,
    mean_adjust_cols: List[str] = None,
    ci_adjust_cols: List[str] = None,
):
    """
    Model the relationship between prediction interval and covariates

    Parameters
    ----------
    df : str
        dataframe
    y : str
        `y` column in the dataframe
    pred : str
        `pred` column in the dataframe
    """


def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()