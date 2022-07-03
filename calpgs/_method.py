import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from typing import List, Dict
from scipy import stats
from scipy.optimize import minimize


def multiple_logpdfs(x, means, covs):
    """
    From http://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/

    x: np.ndarray
        n x d
    means: np.ndarray
        n x d
    covs: np.ndarray
        n x d x d
    """
    # Thankfully, NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs = 1.0 / vals

    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us = vecs * np.sqrt(valsinvs)[:, None]
    devs = x - means

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs = np.einsum("ni,nij->nj", devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas = np.sum(np.square(devUs), axis=1)

    # Compute and broadcast scalar normalizers.
    dim = len(vals[0])
    log2pi = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + mahas + logdets)


def fit_het_linear2(
    y: np.ndarray,
    mean_covar: np.ndarray,
    var_covar: np.ndarray,
    method: str = "remlscore",
    return_est_covar: bool = False,
):
    """
    NOT RECOMMENDED TO USE ANYMORE.
    Fit a linear regression which allows for heterogeneity in variance

    y ~ N(mean_covar * mean_beta, exp(var_covar * var_beta))

    Parameters
    ----------
    y : np.ndarray
        response variables (n_indiv, )
    mean_covar : np.ndarray
        mean covariates (n_indiv, n_covar)
        intercept should be manually added
    var_covar : np.ndarray
        variance covariates (n_indiv, n_covar)
        intercept should be manually added
    method: str
        use which implementation
        - remlscore (default): use remlscore in statmod package
        - optim: directly optimize liklihood in scipy
    return_est_covar : bool
        if True, return the estimated covariance matrix
        (Default value = False)

    Returns
    -------
    mean_beta : np.ndarray
        mean betas (n_covar, )
    var_beta : np.ndarray
        variance betas (n_covar, )
    """
    assert y.ndim == 1
    assert (mean_covar.ndim == 2) & (var_covar.ndim == 2)

    # convert to np.ndarray when possible
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(mean_covar, pd.DataFrame):
        mean_covar = mean_covar.values
    if isinstance(var_covar, pd.DataFrame):
        var_covar = var_covar.values

    assert (
        isinstance(y, np.ndarray)
        & isinstance(mean_covar, np.ndarray)
        & isinstance(var_covar, np.ndarray)
    ), "y, mean_covar, var_covar must be np.ndarray"

    if method == "remlscore":
        fit = remlscore_wrapper(y=y, X=mean_covar, Z=var_covar)
        if return_est_covar:
            return fit
        else:
            return fit[0], fit[1]
    elif method == "optim":
        assert return_est_covar == False, "return_est_covar is not supported for optim"
        n_indiv = len(y)

        assert (mean_covar.shape[0] == n_indiv) & (var_covar.shape[0] == n_indiv)
        # prepend column of 1
        n_mean_params = mean_covar.shape[1]

        def negloglik(params):
            """Evaluate the negative log-likelihood of the model

            Parameters
            ----------
            params : _type_
                pair of np.ndarray mean_beta, var_beta

            Returns
            -------
            float
                negative log-likelihood
            """
            mean_beta, var_beta = params[:n_mean_params], params[n_mean_params:]
            mean = mean_covar @ mean_beta
            var = np.exp(var_covar @ var_beta)

            return (-1) * stats.norm.logpdf(y, loc=mean, scale=np.sqrt(var)).sum()

        # initialize the mean_beta with regression mean
        # initalize the var_beta with [overall_r2, 0, 0, ...]
        init_mean_model = sm.OLS(y, mean_covar).fit()
        init_mean_beta = init_mean_model.params
        init_var = np.mean(init_mean_model.resid ** 2)
        init_var_beta = np.array([init_var] + [0.0] * (var_covar.shape[1] - 1))
        model = minimize(
            negloglik,
            np.concatenate([init_mean_beta, init_var_beta]),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        params = model.x
        mean_beta, var_beta = params[:n_mean_params], params[n_mean_params:]
        return mean_beta, var_beta


def remlscore_wrapper(y: np.ndarray, X: np.ndarray, Z: np.ndarray):
    """
    Wrapper for remlscore function in R.

    Parameters
    ----------
    y : np.ndarray
        response variable.
    X : np.ndarray
        design matrix for mean effects
    Z : np.ndarray
        design matrix for variance effects

    Returns
    -------
    beta: np.ndarray
        estimated mean effects
    gamma: np.ndarray
        estimated gamma effects
    beta_cov: np.ndarray
        estimated covariance matrix of mean effects
    gamma_cov: np.ndarray
        estimated covariance matrix of gamma effects
    """
    if getattr(remlscore_wrapper, "remlscore", None) is None:
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri as numpy2ri

        # import rpy2.robjects as ro

        # ro.conversion.py2ri = numpy2ri

        numpy2ri.activate()
        statmod = importr("statmod")
        remlscore_wrapper.remlscore = statmod.remlscore  # type: ignore

    assert y.ndim == 1
    assert X.ndim == 2
    assert Z.ndim == 2
    assert X.shape[0] == Z.shape[0]
    assert len(y) == X.shape[0]

    assert (
        isinstance(y, np.ndarray)
        & isinstance(X, np.ndarray)
        & isinstance(Z, np.ndarray)
    ), "y, X, Z must be np.ndarray"

    fit = remlscore_wrapper.remlscore(y.reshape(-1, 1), X, Z, tol=1e-6, maxit=100)  # type: ignore
    beta = fit.rx2("beta")
    gamma = fit.rx2("gamma")
    beta_cov = fit.rx2("cov.beta")
    gamma_cov = fit.rx2("cov.gam")
    return beta.flatten(), gamma.flatten(), beta_cov, gamma_cov


def fit_het_linear(
    y: np.ndarray,
    mean_covar: np.ndarray,
    var_covar: np.ndarray,
    slope_covar: np.ndarray = None,
    return_est_covar: bool = False,
):
    """
    Fit a linear regression which allows for heterogeneity in variance

    y ~ N(
          mean=(mean_covar * mean_beta) * (1 + slope_covar * slope_beta),
          var=exp(var_covar * var_beta)
        )

    Parameters
    ----------
    y : np.ndarray
        response variables (n_indiv, )
    mean_covar : np.ndarray
        mean covariates (n_indiv, n_covar)
        all `1` intercept should be manually added
    var_covar : np.ndarray
        variance covariates (n_indiv, n_covar)
        all `1` intercept should be manually added
    slope_covar: np.ndarray
        all `1` intercept should NOT be present
    return_est_covar : bool
        if True, return the estimated covariance matrix
        (Default value = False)

    Returns
    -------
    mean_beta : np.ndarray
        mean betas (n_covar, )
    var_beta : np.ndarray
        variance betas (n_covar, )
    """
    if getattr(fit_het_linear, "r_ext", None) is None:
        from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
        import rpy2.robjects.numpy2ri as numpy2ri

        numpy2ri.activate()

        ext_path = os.path.join(os.path.dirname(__file__), "ext.R")
        with open(ext_path) as f:
            string = "".join(f.readlines())

        fit_het_linear.r_ext = SignatureTranslatedAnonymousPackage(string, "r_ext")  # type: ignore

    # format data
    assert y.ndim == 1
    assert (mean_covar.ndim == 2) & (var_covar.ndim == 2)

    # convert to np.ndarray when possible
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(mean_covar, pd.DataFrame):
        mean_covar = mean_covar.values
    if isinstance(var_covar, pd.DataFrame):
        var_covar = var_covar.values
    if isinstance(slope_covar, pd.DataFrame):
        slope_covar = slope_covar.values

    assert (
        isinstance(y, np.ndarray)
        & isinstance(mean_covar, np.ndarray)
        & isinstance(var_covar, np.ndarray)
    ), "y, mean_covar, var_covar must be np.ndarray"

    if slope_covar is None:
        fit = fit_het_linear.r_ext.fit_het_linear(  # type: ignore
            y=y.reshape(-1, 1), mean_covar=mean_covar, var_covar=var_covar
        )

        if return_est_covar:
            return (
                fit.rx2("mean_beta"),
                fit.rx2("var_beta"),
                fit.rx2("mean_beta_varcov"),
                fit.rx2("var_beta_varcov"),
            )
        else:
            return fit.rx2("mean_beta"), fit.rx2("var_beta")
    else:
        fit = fit_het_linear.r_ext.fit_het_linear(  # type: ignore
            y=y.reshape(-1, 1),
            mean_covar=mean_covar,
            var_covar=var_covar,
            slope_covar=slope_covar,
        )
        if return_est_covar:
            return (
                fit.rx2("mean_beta"),
                fit.rx2("var_beta").flatten(),
                fit.rx2("slope_beta"),
                fit.rx2("mean_beta_varcov"),
                fit.rx2("var_beta_varcov"),
                fit.rx2("slope_beta_varcov"),
            )
        else:
            return (
                fit.rx2("mean_beta"),
                fit.rx2("var_beta").flatten(),
                fit.rx2("slope_beta"),
            )


def calibrate_and_adjust(
    train_mean_covar: np.ndarray,
    train_var_covar: np.ndarray,
    train_y: np.ndarray,
    test_mean_covar: np.ndarray,
    test_var_covar: np.ndarray,
    train_slope_covar: np.ndarray = None,
    test_slope_covar: np.ndarray = None,
):
    """
    Perform the calibration and adjust
    All `1` intercept should be included throughout

    Parameters
    ----------
    train_x : np.ndarray
        (n_indiv, n_mean_cov)
    train_z : np.ndarray
        (n_indiv, n_var_cov)
    train_y : np.ndarray
        (n_indiv, ) phenotype
    test_x : np.ndarray
        (n_indiv, n_mean_cov)
    test_z : np.ndarray
        (n_indiv, n_var_cov)
    """
    assert (train_slope_covar is None) == (test_slope_covar is None)
    fit_slope = train_slope_covar is not None
    fit = fit_het_linear(
        y=train_y,
        mean_covar=train_mean_covar,
        var_covar=train_var_covar,
        slope_covar=train_slope_covar,
        return_est_covar=False,
    )
    if fit_slope:
        mean_beta, var_beta, slope_beta = fit
        pred_mean = test_mean_covar.dot(mean_beta) * (
            1 + test_slope_covar.dot(slope_beta)  # type: ignore
        )
        pred_std = np.sqrt(np.exp(test_var_covar.dot(var_beta)))
        return pred_mean, pred_std, mean_beta, var_beta, slope_beta
    else:
        mean_beta, var_beta = fit
        pred_mean = test_mean_covar.dot(mean_beta)
        pred_std = np.sqrt(np.exp(test_var_covar.dot(var_beta)))
        return pred_mean, pred_std, mean_beta, var_beta