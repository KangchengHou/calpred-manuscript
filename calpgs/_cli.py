import fire
from typing import Union, List
import pandas as pd
import numpy as np
import structlog
from ._evaluate import summarize_pred
import calpgs
import statsmodels.api as sm
from scipy import stats

logger = structlog.get_logger()


def log_params(name, params):
    logger.info(
        f"Received parameters: \n{name}\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in params.items())
    )


def group_stats(
    df,
    y: str,
    pred: str,
    group: Union[str, List[str]],
    out: str,
    predstd: str = None,
    cor: str = "pearson",
    n_subgroup: int = 5,
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
        output prefix, <out>.r2.tsv, <out>.r2diff.tsv will be created
    predstd : str
        Name of the column containing the initial predicted standard deviation.
    cor : str
        Correlation method to use. Options: pearson (default) or spearman.
    n_bootstrap : int
        Number of bootstraps to perform, default 1000.
    """

    np.random.seed(seed)
    log_params("group-r2", locals())
    df = pd.read_csv(df, sep="\t", index_col=0)
    n_raw = df.shape[0]
    df.dropna(subset=[y, pred], inplace=True)
    logger.info(f"{df.shape[0]}/{n_raw} rows without missing values at {y} and {pred}")
    if isinstance(group, str):
        group = [group]

    df_r2 = []
    df_diff = []
    df_cat = []
    df_predint = []

    for col in group:
        # drop the entire row if one of col, y, pred is missing
        subset_cols = [col, y, pred]
        if predstd is not None:
            subset_cols.append(predstd)
        df_tmp = df[subset_cols].dropna()
        n_unique = len(np.unique(df_tmp[col].values))
        if n_unique > n_subgroup:
            logger.info(f"Converting column '{col}' to {n_subgroup} quintiles")
            cat_var = pd.qcut(df_tmp[col], q=n_subgroup, duplicates="drop")
            df_col_cat = pd.DataFrame(
                enumerate(cat_var.cat.categories), columns=["q", "cat"]
            )
            df_col_cat.insert(0, "group", col)
            df_cat.append(df_col_cat)
            df_tmp[col] = cat_var.cat.codes
        df_res, df_res_se, r2_diff = summarize_pred(
            df_tmp,
            y_col=y,
            pred_col=pred,
            predstd_col=predstd,
            group_col=col,
            n_bootstrap=n_bootstrap,
            cor=cor,
            return_r2_diff=True,
        )
        if predstd is not None:
            df_predint.append(
                pd.DataFrame(
                    {
                        "group": col,
                        "subgroup": df_res.index.values,
                        "coverage": df_res["coverage"].values,
                        "coverage_se": df_res_se["coverage"].values,
                        "length": df_res["length"].values,
                        "length_se": df_res_se["length"].values,
                    }
                )
            )
        df_r2.append(
            pd.DataFrame(
                {
                    "group": col,
                    "subgroup": df_res.index.values,
                    "r2": df_res["r2"].values,
                    "r2_se": df_res_se["r2"].values,
                }
            )
        )

        df_diff.append(
            [
                col,
                df_res["r2"].iloc[-1] - df_res["r2"].iloc[0],
                np.mean(r2_diff > 0),
                np.mean(r2_diff) / np.std(r2_diff),
            ]
        )

    pd.concat(df_r2).to_csv(out + ".r2.tsv", sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(df_diff, columns=["group", "r2diff", "prob>0", "zscore"]).to_csv(
        out + ".r2diff.tsv", sep="\t", index=False, float_format="%.6g"
    )
    if len(df_cat) > 0:
        pd.concat(df_cat).to_csv(out + ".cat.tsv", sep="\t", index=False)
    if len(df_predint) > 0:
        pd.concat(df_predint).to_csv(
            out + ".predint.tsv", sep="\t", index=False, float_format="%.6g"
        )


def model(
    df: str,
    out: str,
    y: str = None,
    mean_covar: List[str] = None,
    var_covar: List[str] = None,
    slope_covar: List[str] = None,
    fit_intercept: bool = False,
    verbose: bool = False,
):
    """
    Model the relationship between prediction interval and covariates

    Parameters
    ----------
    df : str
        Path to the dataframe containing the data, deliminated by '\t'. Each row
        represents information for a single individual. Each row starts with the
        individual ID, and the remaining columns are phenotype and covariates. By
        default, the 1st column is individual ID, the 2nd column is phenotype, and
        all remaining columns are covariates; all covariates are used to fit
        mean, variance, and slope. Alternatively, use `--mean-covar`, `--var-covar`,
        and `--slope-covar` to specify the columns to use for mean, variance, and
        slope. All `1` constant values will be automatically handled.
    out : str
        output file, a tab deliminated file will be created
    y : str
        Name of the column containing the observed phenotype. If not specified,
        the 1st column is used.
    mean_covar : List[str]
        Names of the columns containing the mean covariates, separated by `,`.
        If not specified, all covariates columns are used.
    var_covar : List[str]
        Names of the columns containing the variance covariates, separated by `,`.
        If not specified, all covariates columns are used.
    slope_covar : List[str]
        Names of the columns containing the slope covariates, separated by `,`.
        If not specified, all covariates columns are used.
    """
    log_params("model", locals())

    # inputs
    df_data = pd.read_csv(df, sep="\t", index_col=0)

    logger.info(f"Load {df_data.shape[0]} individuals and {df_data.shape[1]} columns")

    if y is None:
        y_col = df_data.columns[0]
    else:
        y_col = y
    if mean_covar is None:
        mean_covar_cols = [col for col in df_data.columns if col != y_col]
    else:
        mean_covar_cols = mean_covar
    if var_covar is None:
        var_covar_cols = [col for col in df_data.columns if col != y_col]
    else:
        var_covar_cols = var_covar
    if slope_covar is None:
        slope_covar_cols = [col for col in df_data.columns if col != y_col]
    else:
        slope_covar_cols = slope_covar

    mean_covar_vals, var_covar_vals, slope_covar_vals = (
        sm.add_constant(df_data[mean_covar_cols]),
        sm.add_constant(df_data[var_covar_cols]),
        df_data[slope_covar_cols],
    )
    logger.info(f"Phenotype = '{y_col}'")

    for name, mat in [
        ("mean", mean_covar_vals),
        ("variance", var_covar_vals),
        ("slope", slope_covar_vals),
    ]:
        logger.info(f"Estimating {name} using {','.join(mat.columns)}")

    fit = calpgs.fit_het_linear(
        y=df_data[y_col].values,
        mean_covar=mean_covar_vals,
        var_covar=var_covar_vals,
        slope_covar=slope_covar_vals,
        return_est_covar=True,
        fit_intercept=fit_intercept,
        trace=verbose,
    )
    if fit_intercept:
        (
            mean_coef,
            var_coef,
            slope_coef,
            intercept_coef,
            mean_vcov,
            var_cov,
            slope_vcov,
            intercept_vcov,
        ) = fit
    else:
        mean_coef, var_coef, slope_coef, mean_vcov, var_cov, slope_vcov = fit
    mean_se = np.sqrt(np.diag(mean_vcov))
    var_se = np.sqrt(np.diag(var_cov))
    slope_se = np.sqrt(np.diag(slope_vcov))

    df_mean_params = pd.DataFrame(
        {"mean_coef": mean_coef, "mean_se": mean_se, "mean_z": mean_coef / mean_se},
        index=mean_covar_vals.columns,
    )
    df_var_params = pd.DataFrame(
        {"var_coef": var_coef, "var_se": var_se, "var_z": var_coef / var_se},
        index=var_covar_vals.columns,
    )
    df_slope_params = pd.DataFrame(
        {
            "slope_coef": slope_coef,
            "slope_se": slope_se,
            "slope_z": slope_coef / slope_se,
        },
        index=slope_covar_vals.columns,
    )

    df_params = pd.concat([df_mean_params, df_var_params, df_slope_params], axis=1)
    if fit_intercept:
        print(intercept_coef, intercept_vcov)
        intercept_se = np.sqrt(np.diag(intercept_vcov))
        df_intercept_params = pd.DataFrame(
            {"intercept_coef": intercept_coef, "intercept_se": intercept_se},
            index=["const"],
        )
        df_params = pd.concat([df_params, df_intercept_params], axis=1)

    df_params.index.name = "param"
    logger.info("Estimated parameters:")
    print(df_params)
    logger.info(f"Writing model to '{out}'")
    df_params.to_csv(out, sep="\t", index=True, float_format="%.6g", na_rep="NA")


def predict(model: str, df: str, out: str, ci: float = 0.9):
    """
    Adjust prediction and prediction standard deviation based on the estimated model

    Parameters
    ----------
    model : str
        Path to the estimated parameter files generated with `calpgs model`
    df : str
        Path to the test file, must contain the same columns as the training file used
        in `calpgs model`
    out : str
        Path to the output file.
    """
    log_params("predict", locals())

    # inputs
    df_data = pd.read_csv(df, sep="\t", index_col=0)
    assert "const" not in df_data.columns, "'const' column in 'df'"
    df_data["const"] = 1.0

    df_params = pd.read_csv(model, sep="\t", index_col=0)
    logger.info(f"Load model parameters:")
    print(df_params)

    mean_coef = df_params["mean_coef"].dropna()
    var_coef = df_params["var_coef"].dropna()
    slope_coef = df_params["slope_coef"].dropna()

    pred_slope = 1 + np.dot(df_data[slope_coef.index].values, slope_coef)
    pred_mean = np.dot(df_data[mean_coef.index].values, mean_coef) * pred_slope
    if "intercept_coef" in df_params.columns:
        intercept_coef = df_params["intercept_coef"].dropna()
        pred_mean += np.dot(df_data[intercept_coef.index].values, intercept_coef)
    pred_std = np.sqrt(np.exp(np.dot(df_data[var_coef.index].values, var_coef)))
    ci_z = stats.norm.ppf((1 + ci) / 2)
    df_pred = pd.DataFrame(
        {
            "mean": pred_mean,
            "std": pred_std,
            "lower": pred_mean - pred_std * ci_z,
            "upper": pred_mean + pred_std * ci_z,
        },
        index=df_data.index,
    )
    logger.info(f"Prediction for first 5 individuals:")
    print(df_pred.head())
    logger.info(f"Writing prediction to '{out}'")
    df_pred.to_csv(out, sep="\t", index=True, float_format="%.6g", na_rep="NA")


def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()
