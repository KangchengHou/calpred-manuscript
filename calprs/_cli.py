import fire
from typing import Union, List
import pandas as pd
import numpy as np
import structlog
from ._evaluate import summarize_pred
import pickle
import calprs


logger = structlog.get_logger()


def log_params(name, params):
    logger.info(
        f"Received parameters: \n{name}\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in params.items())
    )


def r2diff(
    df: str,
    y: str,
    pred: str,
    group: Union[str, List[str]],
    out: str,
    n_bootstrap: int = 1000,
    seed=1234,
):
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

    np.random.seed(seed)
    log_params("r2diff", locals())
    df = pd.read_csv(df, sep="\t", index_col=0)
    if isinstance(group, str):
        group = [group]

    out_list = []
    for col in group:
        n_unique = len(np.unique(df[col].dropna().values))
        if n_unique > 5:
            logger.info(f"Converting column '{col}' to 5 quintiles")
            df[col] = pd.qcut(df[col], q=5, duplicates="drop").cat.codes
        df_res, df_res_se, r2_diff = summarize_pred(
            df,
            y_col=y,
            pred_col=pred,
            group_col=col,
            n_bootstrap=n_bootstrap,
            return_r2_diff=True,
        )
        out_list.append(
            [col, df_res["r2"].iloc[-1] - df_res["r2"].iloc[0], np.mean(r2_diff > 0)]
        )

    df_out = pd.DataFrame(out_list, columns=["group", "r2diff", "prob>0"])
    df_out.to_csv(out, sep="\t", index=False, float_format="%.6g")
    print(df_out)


def model(
    df: str,
    y: str,
    pred: str,
    predstd: str,
    out: str,
    ci: float = 0.9,
    ci_method: str = None,
    mean_adjust_cols: List[str] = None,
    ci_adjust_cols: List[str] = None,
):
    """
    Model the relationship between prediction interval and covariates

    Parameters
    ----------
    df : str
        Path to the dataframe containing the data.
    y : str
        Name of the column containing the observed phenotype.
    pred : str
        Name of the column containing the predicted value.
    predstd : str
        Name of the column containing the initial predicted standard deviation.
    ci: float
        target confidence interval, default 0.9, <lower_col> and <upper_col> will be used for
        calibration
    ci_method: str
        method for calibration, "scale" or "shift"
    mean_adjust_colss: List[str]
        a list of variables to be used for mean adjustment (columns corresponds to
        variables)
    ci_adjust_cols: List[str]
        a list of variables to be used for ci adjustment (columns corresponds to
        variables)
    out: str
        Path to the output pickle file.
    """
    # inputs
    df_train = pd.read_csv(df, sep="\t", index_col=0)

    if isinstance(mean_adjust_cols, str):
        mean_adjust_cols = [mean_adjust_cols]
    if mean_adjust_cols is None:
        mean_adjust_vars = None
    else:
        mean_adjust_vars = df_train[mean_adjust_cols]

    if isinstance(ci_adjust_cols, str):
        ci_adjust_cols = [ci_adjust_cols]
    if ci_adjust_cols is None:
        ci_adjust_vars = None
    else:
        ci_adjust_vars = df_train[ci_adjust_cols]

    if "y" in df_train.columns:
        print("y column is present")
    else:
        print("y column is not here. Popping 'y' will produce KeyError")

    result_model = calprs.calibrate_model(
        y=df_train[y].values,
        pred=df_train[pred].values,
        predstd=df_train[predstd].values,
        ci=ci,
        ci_method=ci_method,
        mean_adjust_vars=mean_adjust_vars,
        ci_adjust_vars=ci_adjust_vars,
    )

    pickle_out = open(out, "wb")
    pickle.dump(result_model, pickle_out)
    pickle_out.close()


def calibrate(
    model: str,
    df: str,
    pred: str,
    predstd: str,
    out: str,
    mean_adjust_cols: List[str] = None,
    ci_adjust_cols: List[str] = None,
):
    """
    Adjust prediction and prediction standard deviation according to calibration model

    Parameters
    ----------
    model : str
        Path to the pickle file containing the model data.
    df : str
        Path to the test data.
    pred : str
        Name of the column containing the predicted value.
    predstd : str
        Name of the column containing the initial predicted standard deviation.
    out : str
        Path to the output file.
    mean_adjust_cols: List[str]
        a list of variables to be used for mean adjustment (columns corresponds to
        variables)
    ci_adjust_cols: List[str]
        a list of variables to be used for ci adjustment (columns corresponds to
        variables)
    """
    # inputs
    pickle_in = open(model, "rb")
    model = pickle.load(pickle_in)
    df_test = pd.read_csv(df, sep="\t", index_col=0)

    if isinstance(mean_adjust_cols, str):
        mean_adjust_cols = [mean_adjust_cols]
    if mean_adjust_cols is None:
        mean_adjust_vars = None
    else:
        mean_adjust_vars = df_test[mean_adjust_cols]

    if isinstance(ci_adjust_cols, str):
        ci_adjust_cols = [ci_adjust_cols]
    if ci_adjust_cols is None:
        ci_adjust_vars = None
    else:
        ci_adjust_vars = df_test[ci_adjust_cols]

    df_test["cal_prs"], df_test["cal_predstd"] = calprs.calibrate_adjust(
        model=model,
        pred=df_test[pred].values,
        predstd=df_test[predstd].values,
        mean_adjust_vars=mean_adjust_vars,
        ci_adjust_vars=ci_adjust_vars,
    )
    df_test.to_csv(out, sep="\t", index=False)


def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()
