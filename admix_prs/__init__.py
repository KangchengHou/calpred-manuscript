import numpy as np
from typing import List, Tuple
import dask.array as da
import dask
import pandas as pd
from scipy import linalg
from tqdm import tqdm
import admix
import dapgen
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from ._utils import *


def simulate_quant_pheno(
    bfile_list, group_list, hsq, causal_prop, freq_file, out_prefix, n_sim=10
):
    assert len(bfile_list) == len(group_list)
    geno = []
    df_indiv = []
    df_snp = None

    for bfile, group in zip(bfile_list, group_list):
        this_geno, this_df_snp, this_df_indiv = dapgen.read_bfile(bfile, snp_chunk=5120)
        if df_snp is None:
            df_snp = this_df_snp
        else:
            assert df_snp.equals(
                this_df_snp
            ), ".bim should be consistent for all bfiles"
        geno.append(this_geno)
        this_df_indiv["GROUP"] = group
        df_indiv.append(this_df_indiv)

    geno = da.concatenate(geno, axis=1)
    df_indiv = pd.concat(df_indiv)

    assert (
        len(set(df_indiv.index)) == df_indiv.shape[0]
    ), "df_indiv.index should be unique"

    # form SNP files
    df_freq = (
        pd.read_csv(freq_file, delim_whitespace=True)
        .set_index("ID")
        .reindex(df_snp.index.values)
    )
    df_snp["FREQ"] = df_freq["ALT_FREQS"].values
    assert df_snp["FREQ"].isna().any() == False

    snp_prior_var = admix.data.calc_snp_prior_var(df_snp, "gcta")

    n_snp, n_indiv = geno.shape
    n_causal = int(n_snp * causal_prop)

    sim = admix.simulate.quant_pheno_1pop(
        geno=geno, hsq=hsq, n_causal=n_causal, n_sim=n_sim, snp_prior_var=snp_prior_var
    )

    df_beta = pd.DataFrame(
        sim["beta"], columns=[f"SIM_{i}" for i in range(n_sim)], index=df_snp.index
    )
    df_beta = pd.concat([df_snp, df_beta], axis=1)

    df_pheno_g = pd.DataFrame(
        sim["pheno_g"], columns=[f"SIM_{i}" for i in range(n_sim)], index=df_indiv.index
    )
    df_pheno_g = pd.concat([df_indiv, df_pheno_g], axis=1)

    df_pheno = pd.DataFrame(
        sim["pheno"], columns=[f"SIM_{i}" for i in range(n_sim)], index=df_indiv.index
    )
    df_pheno = pd.concat([df_indiv, df_pheno], axis=1)

    for suffix, df in zip(
        ["beta", "pheno_g", "pheno"], [df_beta, df_pheno_g, df_pheno]
    ):
        df.to_csv(f"{out_prefix}.{suffix}.tsv", sep="\t")


def calc_prs(bfile, df_weights, n_sample=500):

    geno, df_snp, df_indiv = dapgen.read_bfile(bfile)

    common_snps = set(df_snp.index) & set(df_weights.index)
    print(f"common snp proportion: {len(common_snps) / len(df_snp)}")
    geno_mask = df_snp.index.isin(common_snps)
    weight_mask = df_weights.index.isin(common_snps)

    # subset
    geno = geno[geno_mask, :]
    df_snp = df_snp.loc[geno_mask, :]
    df_weights = df_weights.loc[weight_mask, :]

    assert np.all(
        df_snp[["CHROM", "POS", "REF", "ALT"]].values
        == df_weights[["CHROM", "POS", "REF", "ALT"]].values
    )
    pred = admix.data.geno_mult_mat(
        geno, df_weights[[f"SAMPLE_{i}" for i in range(1, n_sample + 1)]].values
    )

    df_pred = pd.DataFrame(
        pred,
        columns=[f"SAMPLE_{i}" for i in range(1, n_sample + 1)],
        index=df_indiv.index,
    )
    df_pred.insert(0, "MEAN", df_pred.mean(axis=1))
    return df_pred


def calibrate_prs(
    df, idx_cal, mean_cov_cols, q=0.1, method=None, quantile_cov_cols=None
):
    """
    Perform calibration:
    Step 1: fit PHENO ~ PRS_MEAN + intercept + age + sex + 10PCs
    Step 2: Use PRS's interval as seed interval
    Step 3: try scaling or addition

    Parameters
    ---------
    df: pd.DataFrame
        containing PHENO, PRS_MEAN, PRS_Q_{q}, and other covariates
    idx_cal: list
        list of index of individuals used for training calibration
    mean_cov_cols: list
        list of covariates used for calibrating the mean
    quantile_cov_cols: list
        list of covariates used for calibrating the quantile
    q: float
        target quantile, default 0.1, PRS_Q_{q} and PRS_Q_{1-q} will be used for calibration
    method: str
        method for calibration, "scale" or "shift"

    Comments
    --------
    The following is useful for plotting:
    q = 0.1
    df_cal = df_summary
    # model = sm.OLS(
    #     df_cal["PHENO"],
    #     sm.add_constant(df_cal[["PRS_MEAN"] + cov_cols]),
    # ).fit()

    # df_cal["CAL_SCALE"] = np.abs(df_cal["PHENO"] - model.fittedvalues) / (
    #     (df_cal[f"PRS_Q_{1 - q}"] - df_cal[f"PRS_Q_{q}"]) / 2
    # )

    # sns.scatterplot(data=df_cal, x="PC1", y="CAL_SCALE", hue="GROUP")

    # model = smf.quantreg("CAL_SCALE ~ 1 + PC1 + PC2", df_cal).fit(q=0.8)
    # df_cal["FITTED_CAL_SCALE"] = (
    #     model.params["Intercept"]
    #     + model.params["PC1"] * df_cal["PC1"]
    #     + model.params["PC2"] * df_cal["PC2"]
    # )
    # plt.plot(df_cal["PC1"], model.params["Intercept"] + model.params["PC1"] * df_cal["PC1"] + )

    # df_cal.groupby("GROUP").apply(lambda x: np.quantile(x["CAL_SCALE"], 0.8))
    # df_cal.groupby("GROUP").apply(lambda x: np.quantile(x["FITTED_CAL_SCALE"], 0.8))
    """
    assert method in ["scale", "shift"] or method is None
    assert np.all([col in df.columns for col in ["PHENO", "PRS_MEAN"]])
    df = df.copy()
    if quantile_cov_cols is None:
        quantile_cov_cols = []

    df_cal = df.loc[idx_cal].dropna(
        subset=["PHENO"] + mean_cov_cols + quantile_cov_cols
    )
    # step 1: build prediction model
    mean_model = sm.OLS(
        df_cal["PHENO"],
        sm.add_constant(df_cal[["PRS_MEAN"] + mean_cov_cols]),
    ).fit()
    mean_pred = mean_model.predict(sm.add_constant(df[["PRS_MEAN"] + mean_cov_cols]))

    # step 2:
    df_rls = pd.DataFrame(index=df.index)
    if method in ["scale", "shift"]:
        assert q < 0.5, "q should be less than 0.5"
    elif method is None:
        if len(quantile_cov_cols) > 0:
            warnings.warn(
                "`quantile_cov_cols` will not be used because `method` is None"
            )
    else:
        raise ValueError("method should be either scale or shift")

    if method is None:
        # only apply mean shift to the intervals
        df_rls[f"PRS_Q_{q}"] = df[f"PRS_Q_{q}"] - df["PRS_MEAN"] + mean_pred
        df_rls[f"PRS_Q_{1 - q}"] = df[f"PRS_Q_{1 - q}"] - df["PRS_MEAN"] + mean_pred
    elif method == "scale":
        df_cal["CAL_SCALE"] = np.abs(df_cal["PHENO"] - mean_model.fittedvalues) / (
            (df_cal[f"PRS_Q_{1 - q}"] - df_cal[f"PRS_Q_{q}"]) / 2
        )
        interval_len = (df[f"PRS_Q_{1 - q}"] - df[f"PRS_Q_{q}"]) / 2

        if len(quantile_cov_cols) > 0:
            quantreg_model = smf.quantreg(
                "CAL_SCALE ~ 1 + " + "+".join([col for col in quantile_cov_cols]),
                df_cal,
            ).fit(q=(1 - q * 2))

            df["FITTED_CAL_SCALE"] = quantreg_model.params["Intercept"]
            for col in quantile_cov_cols:
                df["FITTED_CAL_SCALE"] += quantreg_model.params[col] * df[col]

            df_rls[f"PRS_Q_{q}"] = mean_pred - interval_len * df["FITTED_CAL_SCALE"]
            df_rls[f"PRS_Q_{1 - q}"] = mean_pred + interval_len * df["FITTED_CAL_SCALE"]
        else:
            cal_scale = np.quantile(
                np.abs(df_cal["PHENO"] - mean_model.fittedvalues)
                / ((df_cal[f"PRS_Q_{1 - q}"] - df_cal[f"PRS_Q_{q}"]) / 2),
                1 - q * 2,
            )
            df_rls[f"PRS_Q_{q}"] = mean_pred - interval_len * cal_scale
            df_rls[f"PRS_Q_{1 - q}"] = mean_pred + interval_len * cal_scale

    elif method == "shift":
        upper = mean_model.fittedvalues + (
            df_cal[f"PRS_Q_{1 - q}"] - df_cal["PRS_MEAN"]
        )
        lower = mean_model.fittedvalues - (df_cal["PRS_MEAN"] - df_cal[f"PRS_Q_{q}"])
        df_cal["CAL_SHIFT"] = np.maximum(
            lower - df_cal["PHENO"], df_cal["PHENO"] - upper
        )

        if len(quantile_cov_cols) > 0:
            quantreg_model = smf.quantreg(
                "CAL_SHIFT ~ 1 + " + "+".join([col for col in quantile_cov_cols]),
                df_cal,
            ).fit(q=(1 - q * 2))

            df["FITTED_CAL_SHIFT"] = quantreg_model.params["Intercept"]
            for col in quantile_cov_cols:
                df["FITTED_CAL_SHIFT"] += quantreg_model.params[col] * df[col]

            df_rls[f"PRS_Q_{q}"] = (
                mean_pred
                - (df[f"PRS_MEAN"] - df[f"PRS_Q_{q}"])
                - df["FITTED_CAL_SHIFT"]
            )

            df_rls[f"PRS_Q_{1 - q}"] = (
                mean_pred
                + (df[f"PRS_Q_{1 - q}"] - df[f"PRS_MEAN"])
                + df["FITTED_CAL_SHIFT"]
            )
        else:
            cal_shift = np.quantile(df_cal["CAL_SHIFT"], 1 - q * 2)
            df_rls[f"PRS_Q_{q}"] = (
                mean_pred - (df[f"PRS_MEAN"] - df[f"PRS_Q_{q}"]) - cal_shift
            )
            df_rls[f"PRS_Q_{1 - q}"] = (
                mean_pred + (df[f"PRS_Q_{1 - q}"] - df[f"PRS_MEAN"]) + cal_shift
            )

    return df_rls

    # quant_cal = df.loc[idx_cal, f"PRS_Q_{q}"]
    # score_cal = quant_cal - df.loc[idx_cal, "PHENO"]
    # correction_cal = np.quantile(score_cal, 1 - q)

    # # predict test
    # quant_test = df.loc[idx_test, f"PRS_Q_{q}"]
    # corrected_test = quant_test - correction_cal
    # df_rls[f"PRS_Q_{q}"] = df[f"PRS_Q_{q}"] - correction_cal
    # df_rls = pd.DataFrame(df_rls, index=df.index)
    # return df_rls.loc[idx_test]
