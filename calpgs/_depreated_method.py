import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import List, Dict
from scipy import stats
from scipy.optimize import minimize

# def calibrate_and_adjust2(
#     train_x: np.ndarray,
#     train_y: np.ndarray,
#     test_x: np.ndarray,
#     ci: float = 0.9,
#     method="qr",
# ):
#     """
#     Perform the calibration and adjust

#     Parameters
#     ----------
#     train_x : np.ndarray
#         (n_indiv, n_cov), intercept should not be included
#     train_y : np.ndarray
#         (n_indiv, ) phenotype
#     test_x : np.ndarray
#         (n_indiv, n_cov) intercept should not be included
#     ci : float
#         target confidence interval, default 0.9, <lower_col> and <upper_col> will be
#         used for calibration
#     method : str
#         method for calibration, 'qr': quantile regression or 'vr': variance regression
#     """
#     ci_z = stats.norm.ppf((1 + ci) / 2)

#     if method == "qr":
#         alpha = (1 - ci) / 2
#         models = [
#             QuantReg(endog=train_y, exog=sm.add_constant(train_x)).fit(q=q)
#             for q in [alpha, 0.5, 1 - alpha]
#         ]
#         preds = [model.predict(sm.add_constant(test_x)) for model in models]
#         pred_median = preds[1]
#         pred_std = (preds[2] - preds[0]) / (2 * ci_z)
#         return pred_median, pred_std
#     elif method == "vr":
#         mean_beta, var_beta = fit_het_linear(
#             y=train_y,
#             mean_covar=sm.add_constant(train_x),
#             var_covar=sm.add_constant(train_x),
#         )

#         pred_mean = sm.add_constant(test_x).dot(mean_beta)
#         pred_std = np.sqrt(np.exp(sm.add_constant(test_x).dot(var_beta)))
#         return pred_mean, pred_std
#     else:
#         raise ValueError("method must be 'qr' or 'vr'")


def het_breuschpagan(resid, exog_het, robust=True):
    r"""
    Breusch-Pagan Lagrange Multiplier test for heteroscedasticity

    The tests the hypothesis that the residual variance does not depend on
    the variables in x in the form

    .. :math: \sigma_i = \sigma * f(\alpha_0 + \alpha z_i)

    Homoscedasticity implies that :math:`\alpha=0`.

    Parameters
    ----------
    resid : array_like
        For the Breusch-Pagan test, this should be the residual of a
        regression. If an array is given in exog, then the residuals are
        calculated by the an OLS regression or resid on exog. In this case
        resid should contain the dependent variable. Exog can be the same as x.
    exog_het : array_like
        This contains variables suspected of being related to
        heteroscedasticity in resid.
    robust : bool, default True
        Flag indicating whether to use the Koenker version of the
        test (default) which assumes independent and identically distributed
        error terms, or the original Breusch-Pagan version which assumes
        residuals are normally distributed.

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue : float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    Assumes x contains constant (for counting dof and calculation of R^2).
    In the general description of LM test, Greene mentions that this test
    exaggerates the significance of results in small or moderately large
    samples. In this case the F-statistic is preferable.

    **Verification**

    Chisquare test statistic is exactly (<1e-13) the same result as bptest
    in R-stats with defaults (studentize=True).

    **Implementation**

    This is calculated using the generic formula for LM test using $R^2$
    (Greene, section 17.6) and not with the explicit formula
    (Greene, section 11.4.3), unless `robust` is set to False.
    The degrees of freedom for the p-value assume x is full rank.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
       5th edition. (2002).
    .. [2]  Breusch, T. S.; Pagan, A. R. (1979). "A Simple Test for
       Heteroskedasticity and Random Coefficient Variation". Econometrica.
       47 (5): 1287–1294.
    .. [3] Koenker, R. (1981). "A note on studentizing a test for
       heteroskedasticity". Journal of Econometrics 17 (1): 107–112.
    """

    x = np.asarray(exog_het)
    y = np.asarray(resid) ** 2
    if not robust:
        y = y / np.mean(y)
    nobs, nvars = x.shape
    resols = sm.OLS(y, x).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared if robust else resols.ess / 2
    # Note: degrees of freedom for LM test is nvars minus constant
    return lm, stats.chi2.sf(lm, nvars - 1), fval, fpval, resols


def test_het_breuschpagan(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    test_col: str,
    control_cols: List[str] = None,
    plot: bool = False,
):
    """
    Test whether a specific covariate influence the predictivity of PRS.
    Using Breusch Pagan test.

    First a model of y ~ pred + test + control
    Then test resid ~ test (resid obtained from the first model)


    Parameters
    ----------
    df: pd.DataFrame
        data frame containing all the information
    y_col: str
        column containing response variable
    pred_col: str
        column containing the mean prediction variable
    test_col: str
        column containing the variable of interest
    control_cols: str
        column(s) containing the the control variables (put into the covariates when performing the control)
        intercept column (all `1') is assumed NOT in the `control_cols`

    Returns
    -------
    pd.Series
        containing p-value, etc.
    het_model: sm.OLS results from resid ** 2 ~ test
    """
    # Fit regression model (using the natural log of one of the regressors)
    y, pred, test = df[y_col], df[[pred_col]], df[[test_col]]
    if control_cols is None:
        control = pd.DataFrame(np.ones((len(df), 1)), columns=["const"], index=df.index)
    else:
        control = sm.add_constant(df[control_cols])
    model = sm.OLS(endog=y, exog=pd.concat([pred, test, control], axis=1)).fit()

    names = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
    # exog_het must always contain a constant
    het_test = het_breuschpagan(resid=model.resid, exog_het=sm.add_constant(test))

    if plot:
        ax = plt.gca()
        ax.scatter(test, model.resid, s=1, alpha=0.05)
        ax.set_xlabel(test_col)
        ax.set_ylabel("Residuals")

    return pd.Series(index=names, data=het_test[0:4]), het_test[4]

def calibrate_pred(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    predstd_col: str,
    calibrate_idx: pd.Index,
    ci: float = 0.9,
    ci_method: str = None,
    mean_adjust_cols: List[str] = None,
    ci_adjust_cols: List[str] = None,
    verbose: bool = False,
):
    """
    Perform calibration:
    Step 1: fit <pheno_col> ~ <pred_col> + intercept + age + sex + 10PCs
    Step 2: Use <lower_col> and <upper_col> interval as seed interval
    Step 3: try scaling or addition

    Parameters
    ---------
    df: pd.DataFrame
        containing <y_col>, <pred_col>, <predstd_col>,
        covariates
    y_col: str
        column name of phenotype
    pred_col: str
        column name of predicted value
    predstd_col: str
        column name of lower bound of interval
    calibrate_idx: list
        list of index of individuals used for training calibration
    mean_adjust_cols: list
        list of covariates used for calibrating the mean
    ci_adjust_cols: list
        list of covariates used for calibrating the quantile
    ci: float
        target quantile, default 0.1, <lower_col> and <upper_col> will be used for
        calibration
    method: str
        method for calibration, "scale" or "shift"
    """
    ci_z = stats.norm.ppf((1 + ci) / 2)
    log = structlog.get_logger()

    assert ci_method in ["scale", "shift"] or ci_method is None
    assert np.all([col in df.columns for col in [y_col, pred_col, predstd_col]])
    df_raw = df.copy()

    if mean_adjust_cols is None:
        mean_adjust_cols = []

    if ci_adjust_cols is None:
        ci_adjust_cols = []

    df_calibrate = df_raw.loc[calibrate_idx].dropna(
        subset=[y_col, pred_col, predstd_col] + mean_adjust_cols + ci_adjust_cols
    )
    # step 1: build prediction model with pheno ~ pred_col + mean_adjust_cols + ...

    mean_model = sm.OLS(
        df_calibrate[y_col],
        sm.add_constant(df_calibrate[[pred_col] + mean_adjust_cols]),
    ).fit()
    # produce prediction for all individuals
    mean_pred = mean_model.predict(
        sm.add_constant(df_raw[[pred_col] + mean_adjust_cols])
    )
    if verbose:
        log.info(
            f"Regress pred_col={pred_col} against "
            f"mean_adjust_cols={mean_adjust_cols} fitted with `calibrate_index` individuals",
        )
        log.info(f"mean_model.summary(): {mean_model.summary()}")

    # step 2:
    df_res = pd.DataFrame(
        {pred_col: mean_pred},
        index=df_raw.index,
    )
    if ci_method in ["scale", "shift"]:
        assert (ci > 0) & (ci < 1), "ci should be between 0 and 1"

    elif ci_method is None:
        if len(ci_adjust_cols) > 0:
            warnings.warn("`ci_adjust_cols` will not be used because `method` is None")
        df_res[predstd_col] = df_raw[predstd_col]
    else:
        raise ValueError("method should be either scale or shift")

    if ci_method is None:
        # only apply mean shift to the intervals
        if verbose:
            log.info(
                f"method=None, no further adjustment to the intervals, only mean shift",
            )
    elif ci_method == "scale":
        if verbose:
            log.info(
                f"method={ci_method}, scale the interval length to the target quantile {ci}"
                f" using quantile_adjust_cols={ci_adjust_cols} with"
                " `calibrate_index` individuals",
            )
        df_calibrate["tmp_scale"] = np.abs(
            df_calibrate[y_col] - mean_model.fittedvalues
        ) / (df_calibrate[predstd_col] * ci_z)

        if len(ci_adjust_cols) > 0:
            # fit a quantile regression model
            quantreg_model = smf.quantreg(
                "tmp_scale ~ 1 + " + "+".join([col for col in ci_adjust_cols]),
                df_calibrate,
            ).fit(q=ci)

            df_raw["tmp_fitted_scale"] = quantreg_model.params["Intercept"]
            for col in ci_adjust_cols:
                df_raw["tmp_fitted_scale"] += quantreg_model.params[col] * df_raw[col]

            df_res[predstd_col] = df_raw[predstd_col] * df_raw["tmp_fitted_scale"]
        else:
            # no variable to fit for quantile regression model
            cal_scale = np.quantile(
                np.abs(df_calibrate[y_col] - mean_model.fittedvalues)
                / (df_calibrate[predstd_col] * ci_z),
                ci,
            )
            df_res[predstd_col] = df_raw[predstd_col] * cal_scale

    elif ci_method == "shift":
        if verbose:
            log.info(
                f"method={ci_method}, expand the interval to the target quantile {ci}"
                f" using quantile_adjust_cols={ci_adjust_cols} with"
                " `calibrate_index` individuals",
            )
        upper = mean_model.fittedvalues + df_calibrate[predstd_col] * ci_z
        lower = mean_model.fittedvalues - df_calibrate[predstd_col] * ci_z
        df_calibrate["tmp_shift"] = np.maximum(
            lower - df_calibrate[y_col], df_calibrate[y_col] - upper
        )
        if len(ci_adjust_cols) > 0:
            quantreg_model = smf.quantreg(
                "tmp_shift ~ 1 + " + "+".join([col for col in ci_adjust_cols]),
                df_calibrate,
            ).fit(q=ci)

            df_raw["tmp_fitted_shift"] = quantreg_model.params["Intercept"]
            for col in ci_adjust_cols:
                df_raw["tmp_fitted_shift"] += quantreg_model.params[col] * df_raw[col]

            df_res[predstd_col] = (
                df_raw[predstd_col] * ci_z + df_raw["tmp_fitted_shift"]
            ) / ci_z
        else:
            cal_shift = np.quantile(df_calibrate["tmp_shift"], ci)
            df_res[predstd_col] = (df_raw[predstd_col] * ci_z + cal_shift) / ci_z
    return df_res


def calibrate_model(
    y: np.ndarray,
    pred: np.ndarray,
    predstd: np.ndarray,
    ci: float = 0.9,
    ci_method: str = None,
    mean_adjust_vars: pd.DataFrame = None,
    ci_adjust_vars: pd.DataFrame = None,
) -> Dict:
    """
    Perform calibration:
    Step 1: fit <pheno_col> ~ <pred_col> + intercept + age + sex + 10PCs
    Step 2: Use <lower_col> and <upper_col> interval as seed interval
    Step 3: try scaling or addition

    Note: NaN values are not allowed in any of the passed data.

    Parameters
    ---------
    y: np.ndarray
        1d array of phenotype
    pred: np.ndarray
        1d array of predicted values
    predstd: np.ndarray
        1d array of prediction standard deviation
    mean_adjust_vars: np.ndarray
        2d array of variables to be used for mean adjustment (columns corresponds to
        variables)
    ci_adjust_vars: np.ndarray
        2d array of variables to be used for ci adjustment (columns corresponds to
        variables)
    ci: float
        target confidence interval, default 0.9, <lower_col> and <upper_col> will be used for
        calibration
    ci_method: str
        method for calibration, "scale" or "shift"

    Returns
    -------
    model: Dict
        mean_model: model
            model to adjust for the mean
        ci_method: str
            method used for calibration, one of None, "scale" or "shift"
        ci_model: float, model
            model to adjust for the confidence interval, None if ci_method is None
            Can be a float, then the adjustment is done the same across individuals.
            Can be a model, then the adjustment is done according to the covariate.

    """
    y, pred, predstd = np.array(y), np.array(pred), np.array(predstd)
    # resulting calibration model
    result_model = dict()
    ci_z = stats.norm.ppf((1 + ci) / 2)

    result_model["ci"] = ci
    result_model["ci_z"] = ci_z

    n_indiv = len(y)
    assert (len(pred) == n_indiv) & (
        len(predstd) == n_indiv
    ), "y, pred, predstd must have the same length (number of individuals)"

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

    # step 1: build prediction model with pheno ~ pred_col + mean_adjust_cols + ...
    mean_design = pd.DataFrame(np.hstack([pred.reshape(-1, 1), mean_adjust_vars]))
    if mean_adjust_vars.shape[1] == 0:
        mean_design.columns = ["pred"]
    else:
        mean_design.columns = ["pred"] + mean_adjust_vars.columns.tolist()  # type: ignore

    mean_model = sm.OLS(
        y,
        sm.add_constant(mean_design),
    ).fit()

    result_model["mean_model"] = mean_model
    pred = mean_model.fittedvalues.values

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
        scale = np.abs(y - pred) / (predstd * ci_z)

        if ci_adjust_vars.shape[1] > 0:
            # fit a quantile regression model
            quantreg_model = QuantReg(
                endog=scale, exog=sm.add_constant(ci_adjust_vars)
            ).fit(q=ci)
            fitted_scale = quantreg_model.fittedvalues
            predstd = predstd * fitted_scale
            result_model["ci_model"] = quantreg_model
        else:
            # use simple conformal prediction
            # because no variable to fit for quantile regression model
            fitted_scale = np.quantile(
                np.abs(y - pred) / (predstd * ci_z),
                ci,
            )
            predstd = predstd * fitted_scale
            result_model["ci_model"] = fitted_scale

    elif ci_method == "shift":
        result_model["ci_method"] = "shift"

        upper = pred + predstd * ci_z
        lower = pred - predstd * ci_z
        shift = np.maximum(lower - y, y - upper)
        if ci_adjust_vars.shape[1] > 0:
            quantreg_model = QuantReg(
                endog=shift, exog=sm.add_constant(ci_adjust_vars)
            ).fit(q=ci)
            fitted_shift = quantreg_model.fittedvalues
            predstd = (predstd * ci_z + fitted_shift) / ci_z
            result_model["ci_model"] = quantreg_model

        else:
            fitted_shift = np.quantile(shift, ci)
            # (1) get 90% CI (2) add a shift (3) scale back
            predstd = (predstd * ci_z + fitted_shift) / ci_z
            result_model["ci_model"] = fitted_shift

    return result_model


def calibrate_adjust(
    model: Dict,
    pred: np.ndarray,
    predstd: np.ndarray,
    mean_adjust_vars: np.ndarray = None,
    ci_adjust_vars: np.ndarray = None,
):
    """
    Adjust prediction and prediction standard deviation according to calibration model

    Parameters
    ---------
    pred: np.ndarray
        1d array of predicted values
    predstd: np.ndarray
        1d array of prediction standard deviation
    model: Dict
        calibration model
    mean_adjust_vars: np.ndarray
        2d array of variables to be used for mean adjustment (columns corresponds to
        variables)
    ci_adjust_vars: np.ndarray
        2d array of variables to be used for ci adjustment (columns corresponds to
        variables)

    Returns
    -------
    pred: np.ndarray
        1d array of predicted values
    predstd: np.ndarray
        1d array of prediction standard deviation
    """

    pred, predstd = np.array(pred), np.array(predstd)
    n_indiv = len(pred)
    assert (len(pred) == n_indiv) & (
        len(predstd) == n_indiv
    ), "pred, predstd must have the same length (number of individuals)"

    if mean_adjust_vars is None:
        mean_adjust_vars = np.zeros([n_indiv, 0])

    if ci_adjust_vars is None:
        ci_adjust_vars = np.zeros([n_indiv, 0])

    # mean adjustment
    pred = model["mean_model"].predict(
        sm.add_constant(np.hstack([pred.reshape(-1, 1), mean_adjust_vars]))
    )

    # ci adjustment
    if model["ci_method"] is None:
        return pred, predstd
    elif model["ci_method"] == "scale":
        if isinstance(model["ci_model"], float):
            assert ci_adjust_vars.shape[1] == 0, "ci_adjust_vars should be empty"
            fitted_scale = model["ci_model"]
        else:
            fitted_scale = model["ci_model"].predict(sm.add_constant(ci_adjust_vars))
        predstd = predstd * fitted_scale

    elif model["ci_method"] == "shift":
        if isinstance(model["ci_model"], float):
            assert ci_adjust_vars.shape[1] == 0, "ci_adjust_vars should be empty"
            fitted_shift = model["ci_model"]
            # (1) get 90% CI (2) add a shift (3) scale back
            ci_z = model["ci_z"]
        else:
            fitted_shift = model["ci_model"].predict(sm.add_constant(ci_adjust_vars))
        predstd = (predstd * ci_z + fitted_shift) / ci_z

    if isinstance(predstd, pd.Series):
        predstd = predstd.values
    return pred, predstd

def estimate_het(xy: np.ndarray, covar: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Parameters
    ----------
    xy: np.ndarray
        n x 2 (x, y) variables, assumed to have variance 1
    covar: np.ndarray
        n x c covariates, assume no intercept is present
    eps: float
        epsilon to prevent rho = 1, which makes the multivariate normal
        degenerate.

    Returns
    -------
    params: np.ndarray
        (c + 1) estimates
    """
    assert xy.ndim == 2
    assert covar.ndim == 2
    # prepend column of 1
    n_data = xy.shape[0]
    covar = np.c_[np.ones(covar.shape[0]), covar]
    n_covar = covar.shape[1]
    assert covar.shape[0] == n_data

    def negloglik(params):
        rho = np.clip(covar @ params, -1 + eps, 1 - eps)

        covs = np.zeros((n_data, 2, 2))
        covs[:, 0, 0] = covs[:, 1, 1] = 1.0
        covs[:, 0, 1] = covs[:, 1, 0] = rho

        return (-1) * multiple_logpdfs(x=xy, means=np.array([0, 0]), covs=covs).sum()

    # initialize the intercept with overall R2
    avg_coef = np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]
    model = minimize(
        negloglik, np.array([avg_coef] + [0.0] * (n_covar - 1)), method="Nelder-Mead"
    )
    return model