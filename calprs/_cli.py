import fire
from typing import Union, List
import pandas as pd
import numpy as np
import structlog
from ._evaluate import summarize_pred
import pickle
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats


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
    mean_adjust_vars: List[str] = None,
    ci_adjust_vars: List[str] = None,
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
    mean_adjust_cols: np.ndarray
        2d array of variables to be used for mean adjustment (columns corresponds to
        variables)
    ci_adjust_cols: np.ndarray
        2d array of variables to be used for ci adjustment (columns corresponds to
        variables)
    out : str
        Path to the output pickle file.
    """
    # inputs
    df_train = pd.read_csv(df, index_col=0)
    y_train = df_train[y].values
    pred_train = df_train[pred].values
    predstd_train = df_train[predstd].values
    n_indiv = len(y_train)
    assert (len(pred_train) == n_indiv) & (
        len(predstd_train) == n_indiv
    ), "y, pred, predstd0 must have the same length (number of individuals)"
    if mean_adjust_vars is None:
        mean_adjust_vars = np.zeros([n_indiv, 0])
    else:
        assert isinstance(
            mean_adjust_vars, pd.DataFrame
        ), "mean_adjust_vars must be a DataFrame"
    if ci_adjust_vars is None:
        ci_adjust_vars = np.zeros([n_indiv, 0])
    else:
        assert isinstance(
            ci_adjust_vars, pd.DataFrame
        ), "ci_adjust_vars must be a DataFrame"
    
    result_model = dict()
    ci_z = stats.norm.ppf((1 + ci) / 2)
    result_model["ci"] = ci
    result_model["ci_z"] = ci_z
    
    # step 1: build prediction model with pheno ~ pred_col + mean_adjust_cols + ...
    mean_design = pd.DataFrame(np.hstack([pred_train.reshape(-1, 1), mean_adjust_vars]))
    if mean_adjust_vars.shape[1] == 0:
        mean_design.columns = ["pred"]
    else:
        mean_design.columns = ["pred"] + mean_adjust_vars.columns.tolist()  # type: ignore

    mean_model = sm.OLS(
        y_train,
        sm.add_constant(mean_design),
    ).fit()

    result_model["mean_model"] = mean_model
    pred_train = mean_model.fittedvalues.values
    
    # step 2: CI calibration
    if ci_method in ["scale", "shift"]:
        assert (ci > 0) & (ci < 1), "ci should be between 0 and 1"
    elif ci_method is None:
        if ci_adjust_vars.shape[1] > 0:
            warnings.warn("`ci_adjust_vars` will not be used because `method` is None")
    else:
        raise ValueError("method should be either scale or shift")
    
    if ci_method is None:
        result_model["ci_method"] = None
        pass
    
    elif ci_method == "scale":
        result_model["ci_method"] = "scale"
        scale = np.abs(y_train - pred_train) / (predstd_train * ci_z)

        if ci_adjust_vars.shape[1] > 0:
            # fit a quantile regression model
            quantreg_model = QuantReg(
                endog=scale, exog=sm.add_constant(ci_adjust_vars)
            ).fit(q=ci)
            fitted_scale = quantreg_model.fittedvalues
            predstd_train *= fitted_scale
            result_model["ci_model"] = quantreg_model
        else:
            # use simple conformal prediction
            # because no variable to fit for quantile regression model
            fitted_scale = np.quantile(
                np.abs(y_train - pred_train) / (predstd_train * ci_z),
                ci,
            )
            predstd_train *= fitted_scale
            result_model["ci_model"] = fitted_scale
            
    elif ci_method == "shift":
        result_model["ci_method"] = "shift"

        upper = pred_train + predstd_train * ci_z
        lower = pred_train - predstd_train * ci_z
        shift = np.maximum(lower - y_train, y_train - upper)
        if ci_adjust_vars.shape[1] > 0:
            quantreg_model = QuantReg(
                endog=shift, exog=sm.add_constant(ci_adjust_vars)
            ).fit(q=ci)
            fitted_shift = quantreg_model.fittedvalues
            predstd_train = (predstd_train * ci_z + fitted_shift) / ci_z
            result_model["ci_model"] = quantreg_model

        else:
            fitted_shift = np.quantile(shift, ci)
            # (1) get 90% CI (2) add a shift (3) scale back
            predstd_train = (predstd_train * ci_z + fitted_shift) / ci_z
            result_model["ci_model"] = fitted_shift
            
    pickle_out = open(out, "wb")
    pickle.dump(result_model, pickle_out)
    pickle_out.close()
    
def calibrate(
    model: str,
    df: str,
    pred: str,
    predstd: str,
    out: str,
    mean_adjust_vars: List[str] = None,
    ci_adjust_vars: List[str] = None,
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
    mean_adjust_cols: np.ndarray
        2d array of variables to be used for mean adjustment (columns corresponds to
        variables)
    ci_adjust_cols: np.ndarray
        2d array of variables to be used for ci adjustment (columns corresponds to
        variables)
    """
    # inputs
    pickle_in = open(model, "rb")
    model = pickle.load(pickle_in)
    df_test = pd.read_csv(df, index_col=0)
    pred_test = df_test[pred].values
    predstd_test = df_test[predstd].values
    n_indiv = len(pred_test)
    assert (len(pred_test) == n_indiv) & (
        len(predstd_test) == n_indiv
    ), "pred, predstd must have the same length (number of individuals)"

    if mean_adjust_vars is None:
        mean_adjust_vars = np.zeros([n_indiv, 0])

    if ci_adjust_vars is None:
        ci_adjust_vars = np.zeros([n_indiv, 0])
    
    # mean adjustment
    pred_test = model["mean_model"].predict(
        sm.add_constant(np.hstack([pred_test.reshape(-1, 1), mean_adjust_vars]))
    )
    
    # ci adjustment
    if model["ci_method"] is None:
        return pred_test, predstd_test
    elif model["ci_method"] == "scale":
        if isinstance(model["ci_model"], float):
            assert ci_adjust_vars.shape[1] == 0, "ci_adjust_vars should be empty"
            fitted_scale = model["ci_model"]
        else:
            fitted_scale = model["ci_model"].predict(sm.add_constant(ci_adjust_vars))
        predstd_test *= fitted_scale

    elif model["ci_method"] == "shift":
        if isinstance(model["ci_model"], float):
            assert ci_adjust_vars.shape[1] == 0, "ci_adjust_vars should be empty"
            fitted_shift = model["ci_model"]
            # (1) get 90% CI (2) add a shift (3) scale back
            ci_z = model["ci_z"]
        else:
            fitted_shift = model["ci_model"].predict(sm.add_constant(ci_adjust_vars))
        predstd_test = (predstd_test * ci_z + fitted_shift) / ci_z
    
    if isinstance(predstd_test, pd.Series):
        predstd_test = predstd_test.values
    df_test["cal_prs"], df_test["cal_predstd"] = pred_test, predstd_test
    df_test.to_csv(out, index=False)



def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()