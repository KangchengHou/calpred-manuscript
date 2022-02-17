import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import warnings
from typing import List
import structlog
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats


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
    true_col: str,
    lower_col: str,
    upper_col: str,
    calibrate_idx: pd.Index,
    q: float = 0.1,
    method: str = None,
    mean_adjust_cols: List[str] = None,
    quantile_adjust_cols: List[str] = None,
):
    """
    Perform calibration:
    Step 1: fit <pheno_col> ~ <pred_col> + intercept + age + sex + 10PCs
    Step 2: Use <lower_col> and <upper_col> interval as seed interval
    Step 3: try scaling or addition

    Parameters
    ---------
    df: pd.DataFrame
        containing <pheno_col>, <pred_col>, <lower_col>, <upper_col>, and other
        covariates
    pheno_col: str
        column name of phenotype
    pred_col: str
        column name of predicted value
    lower_col: str
        column name of lower bound of interval
    upper_col: str
        column name of upper bound of interval
    calibrate_idx: list
        list of index of individuals used for training calibration
    mean_adjust_cols: list
        list of covariates used for calibrating the mean
    quantile_adjust_cols: list
        list of covariates used for calibrating the quantile
    q: float
        target quantile, default 0.1, <lower_col> and <upper_col> will be used for
        calibration
    method: str
        method for calibration, "scale" or "shift"
    """
    log = structlog.get_logger()

    assert method in ["scale", "shift"] or method is None
    assert np.all([col in df.columns for col in [true_col, lower_col, upper_col]])
    df = df.copy()
    pred_col = "PRED"
    assert pred_col not in df.columns
    df[pred_col] = (df[lower_col] + df[upper_col]) / 2

    if mean_adjust_cols is None:
        mean_adjust_cols = []

    if quantile_adjust_cols is None:
        quantile_adjust_cols = []

    df_calibrate = df.loc[calibrate_idx].dropna(
        subset=[true_col, pred_col, lower_col, upper_col]
        + mean_adjust_cols
        + quantile_adjust_cols
    )
    # step 1: build prediction model with pheno ~ pred_col + mean_adjust_cols + ...

    mean_model = sm.OLS(
        df_calibrate[true_col],
        sm.add_constant(df_calibrate[[pred_col] + mean_adjust_cols]),
    ).fit()
    # produce prediction for all individuals
    mean_pred = mean_model.predict(sm.add_constant(df[[pred_col] + mean_adjust_cols]))
    log.info(
        f"Regress pred_col={pred_col} against "
        f"mean_adjust_cols={mean_adjust_cols} fitted with `calibrate_index` individuals",
    )
    log.info(f"mean_model.summary(): {mean_model.summary()}")
    # step 2:
    df_res = pd.DataFrame(
        {"PRED": mean_pred},
        index=df.index,
    )
    if method in ["scale", "shift"]:
        assert q < 0.5, "q should be less than 0.5"

    elif method is None:
        if len(quantile_adjust_cols) > 0:
            warnings.warn(
                "`quantile_adjust_cols` will not be used because `method` is None"
            )
    else:
        raise ValueError("method should be either scale or shift")

    if method is None:
        # only apply mean shift to the intervals
        df_res[lower_col] = df[lower_col] - df[pred_col] + mean_pred
        df_res[upper_col] = df[upper_col] - df[pred_col] + mean_pred
        log.info(
            f"method=None, no further adjustment to the intervals, only mean shift",
        )
    elif method == "scale":
        log.info(
            f"method={method}, scale the interval length to the target quantile {q}"
            f" using quantile_adjust_cols={quantile_adjust_cols} with"
            " `calibrate_index` individuals",
        )
        df_calibrate["tmp_scale"] = np.abs(
            df_calibrate[true_col] - mean_model.fittedvalues
        ) / ((df_calibrate[upper_col] - df_calibrate[lower_col]) / 2)

        interval_len = (df[upper_col] - df[lower_col]) / 2
        if len(quantile_adjust_cols) > 0:
            quantreg_model = smf.quantreg(
                "tmp_scale ~ 1 + " + "+".join([col for col in quantile_adjust_cols]),
                df_calibrate,
            ).fit(q=(1 - q * 2))

            df["tmp_fitted_scale"] = quantreg_model.params["Intercept"]
            for col in quantile_adjust_cols:
                df["tmp_fitted_scale"] += quantreg_model.params[col] * df[col]

            df_res[lower_col] = mean_pred - interval_len * df["tmp_fitted_scale"]
            df_res[upper_col] = mean_pred + interval_len * df["tmp_fitted_scale"]
        else:
            cal_scale = np.quantile(
                np.abs(df_calibrate[true_col] - mean_model.fittedvalues)
                / ((df_calibrate[upper_col] - df_calibrate[lower_col]) / 2),
                1 - q * 2,
            )
            df_res[lower_col] = mean_pred - interval_len * cal_scale
            df_res[upper_col] = mean_pred + interval_len * cal_scale

    elif method == "shift":
        log.info(
            f"method={method}, expand the interval to the target quantile {q}"
            f" using quantile_adjust_cols={quantile_adjust_cols} with"
            " `calibrate_index` individuals",
        )
        upper = mean_model.fittedvalues + (
            df_calibrate[upper_col] - df_calibrate[pred_col]
        )
        lower = mean_model.fittedvalues - (
            df_calibrate[pred_col] - df_calibrate[lower_col]
        )
        df_calibrate["tmp_shift"] = np.maximum(
            lower - df_calibrate[true_col], df_calibrate[true_col] - upper
        )

        if len(quantile_adjust_cols) > 0:
            quantreg_model = smf.quantreg(
                "tmp_shift ~ 1 + " + "+".join([col for col in quantile_adjust_cols]),
                df_calibrate,
            ).fit(q=(1 - q * 2))

            df["tmp_fitted_shift"] = quantreg_model.params["Intercept"]
            for col in quantile_adjust_cols:
                df["tmp_fitted_shift"] += quantreg_model.params[col] * df[col]

            df_res[lower_col] = (
                mean_pred - (df[pred_col] - df[lower_col]) - df["tmp_fitted_shift"]
            )

            df_res[upper_col] = (
                mean_pred + (df[upper_col] - df[pred_col]) + df["tmp_fitted_shift"]
            )
        else:
            cal_shift = np.quantile(df_calibrate["tmp_shift"], 1 - q * 2)
            df_res[lower_col] = mean_pred - (df[pred_col] - df[lower_col]) - cal_shift
            df_res[upper_col] = mean_pred + (df[upper_col] - df[pred_col]) + cal_shift

    return df_res
