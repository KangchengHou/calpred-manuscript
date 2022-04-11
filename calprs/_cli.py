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


def r2diff(df: str, y: str, pred: str, group: Union[str, List[str]], out: str):
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
    """
    log_params("r2diff", locals())
    df = pd.read_csv(df, sep='\t', index_col=0)
    print(df)


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