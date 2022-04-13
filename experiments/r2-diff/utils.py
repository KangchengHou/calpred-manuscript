import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import glob
import statsmodels.api as sm
from scipy.stats import pearsonr
from typing import List
import warnings
import admix_prs

# CONSTANTS
COVAR_COLS = ["AGE", "SEX", "DEPRIVATION_INDEX"] + [f"PC{i}" for i in range(1, 11)]
DATA_DIR = "/u/project/pasaniuc/pasaniucdata/admixture/projects/calprs/experiments/00-compile-data/out"
PHENO_DIR = "/u/project/sgss/UKBB/PRS-RESEARCH/03-compile-pheno/out"


def load_trait_info(
    trait: str, indiv_group: str, covar_cols: List[str]
) -> pd.DataFrame:
    """
    Load
    (1) trait values PHENO_DIR/{trait}.tsv
    (2) polygenic score DATA_DIR/pred/{trait}.score_summary.tsv.gz
    (3) covariates DATA_DIR/covar.tsv
    (3) covariates to adjust for, e.g., age, sex, top 10 PCs
    (4) covariate to test

    Parameters
    ----------
    trait: str
        trait name
    indiv_group: str
        "white_british" or "other"
    covar_cols: List[str]
        list of covariates to load from covar file
    """

    ## 1. load trait and score
    df_trait = pd.read_csv(
        os.path.join(PHENO_DIR, f"{trait}.tsv"), sep="\t", index_col=0
    ).drop(columns=["IID"])

    df_score = pd.read_csv(
        os.path.join(DATA_DIR, f"pred/{trait}.score_summary.tsv.gz"),
        sep="\t",
        index_col=0,
    )
    df_score.index = [int(i.split("_")[0]) for i in df_score.index]

    ## 2. load covariates
    df_covar = pd.read_csv(os.path.join(DATA_DIR, "covar.tsv"), sep="\t", index_col=0)

    # add some phenotype to the covariates
    for col in covar_cols:
        if col in df_covar.columns:
            continue
        else:
            tmp_path = os.path.join(PHENO_DIR, f"{col}.tsv")
            if os.path.exists(tmp_path):
                df_tmp = pd.read_csv(tmp_path, sep="\t", index_col=0).drop(
                    columns=["IID"]
                )
                df_covar[col] = df_tmp["PHENO"].reindex(df_covar.index)
            else:
                warnings.warn(f"{tmp_path} does not exist")

    # merge all files together
    df_trait = pd.merge(df_score, df_trait, left_index=True, right_index=True)
    df_trait = pd.merge(df_trait, df_covar, left_index=True, right_index=True)

    # restricted to indiv_group
    if indiv_group == "white_british":
        df_trait = df_trait[df_trait.group == "United Kingdom"]
    elif indiv_group == "other":
        df_trait = df_trait[~(df_trait.group == "United Kingdom")]
    df_trait = df_trait.dropna()

    return df_trait