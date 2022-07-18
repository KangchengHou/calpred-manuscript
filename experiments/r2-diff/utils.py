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
import calpgs
import matplotlib.transforms as mtrans
import seaborn as sns

# CONSTANTS
COVAR_COLS = ["AGE", "SEX", "DEPRIVATION_INDEX"] + [f"PC{i}" for i in range(1, 11)]
DATA_DIR = "/u/project/pasaniuc/pasaniucdata/admixture/projects/calpgs/experiments/00-compile-data/out"
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
    if indiv_group == "white":
        df_trait = df_trait[df_trait.group == "United Kingdom"]
    elif indiv_group == "other":
        df_trait = df_trait[~(df_trait.group == "United Kingdom")]
    df_trait = df_trait.dropna()

    return df_trait

def load_r2_result(result_dir, group):
    """
    Load 
    (1) <result_dir>/baseline_r2.<group>.tsv
    (2) <result_dir>/r2diff.<group>.tsv
    
    Returns
    -------
    df_baseline : pd.DataFrame
        Baseline R2 evaluated across all individuals
    df_diff: pd.DataFrame
        difference of R2 across 2 covariate groups
    df_diff_pval: pd.DataFrame
        two-sided p-value for R2 difference = 0
    df_diff_zscore: pd.DataFrame
        z-score
    """
    
    df_baseline = pd.read_csv(
        f"{result_dir}/baseline_r2.{group}.tsv", sep="\t", index_col=0
    )["baseline_r2"]

    # read R2 diff and p-values
    df_tmp = pd.read_csv(f"{result_dir}/r2diff.{group}.tsv", sep="\t")
    df_tmp["std_r2diff"] = df_tmp.apply(lambda r: r.r2diff / df_baseline[r.trait], axis=1)

    df_diff = df_tmp.pivot(index="group", columns="trait", values="std_r2diff")
    df_diff_pval = df_tmp.pivot(index="group", columns="trait", values="prob>0")
    df_diff_pval = df_diff_pval.applymap(lambda x: 2 * min(x, 1 - x))
    df_diff_zscore = df_tmp.pivot(index='group', columns="trait", values="zscore")
    df_diff, df_diff_pval, df_diff_zscore = (
        df_diff[df_baseline.index.values],
        df_diff_pval[df_baseline.index.values],
        df_diff_zscore[df_baseline.index.values]
    )

    # p-value are wrongly assigned for some trait, cov pair with 0 R2 difference
    df_diff_pval[df_diff == 0] = 1.0
    df_diff_zscore[df_diff == 0] = 0.0
    

    assert np.all(df_baseline.index == df_diff.columns)
    assert np.all(df_baseline.index == df_diff_pval.columns)
    assert np.all(df_baseline.index == df_diff_zscore.columns)
    
    
    return df_baseline, df_diff, df_diff_pval, df_diff_zscore

def plot_heatmap(
    df_val,
    df_annot=None,
    annot_kws=None,
    cmap="RdBu_r",
    dpi=150,
    squaresize=20,
    heatmap_linewidths=0.5,
    heatmap_linecolor="gray",
    heatmap_xticklabels=True,
    heatmap_yticklabels=True,
    heatmap_cbar=True,
    heatmap_cbar_kws=dict(use_gridspec=False, location="top", fraction=0.03, pad=0.01),
    heatmap_vmin=-5,
    heatmap_vmax=5,
    xticklabels_rotation=45,
    colormap_n_bin=10,
):
    """Plot heatmap
    """
    figwidth = df_val.shape[1] * squaresize / float(dpi)
    figheight = df_val.shape[0] * squaresize / float(dpi)
    fig, ax = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    sns.heatmap(
        df_val,
        cmap=cmap,
        linewidths=heatmap_linewidths,
        linecolor=heatmap_linecolor,
        square=True,
        annot=df_annot,
        annot_kws=annot_kws,
        fmt="",
        ax=ax,
        xticklabels=heatmap_xticklabels,
        yticklabels=heatmap_yticklabels,
        cbar=heatmap_cbar,
        cbar_kws=heatmap_cbar_kws,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    plt.yticks(fontsize=8)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=xticklabels_rotation,
        va="top",
        ha="right",
        fontsize=8,
    )
    ax.tick_params(left=False, bottom=False, pad=-2)
    trans = mtrans.Affine2D().translate(5, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)
    return fig, ax

def plot_heatmap2(
    df_val: pd.DataFrame,
    df_size: pd.DataFrame,
    cmap,
    squaresize: int = 45,
    dpi=150,
    xticklabels_rotation=45,
):

    # manipulate data
    assert np.all(df_val.index.values == df_size.index.values)
    assert np.all(df_val.columns.values == df_size.columns.values)

    n_row, n_col = df_val.shape

    df_val = pd.melt(df_val.T.reset_index(), id_vars=df_val.columns.name)
    df_size = pd.melt(df_size.T.reset_index(), id_vars=df_size.columns.name)

    df_plot = pd.DataFrame(
        {
            "x": df_val.iloc[:, 0],
            "y": df_val.iloc[:, 1],
            "val": df_val.iloc[:, 2],
            "size": df_size.iloc[:, 2],
        }
    )

    figwidth = n_col * squaresize / float(dpi)
    figheight = n_row * squaresize / float(dpi)

    fig, ax = plt.subplots(figsize=(figwidth, figheight), dpi=dpi)

    ax.set_xlim(-0.5, n_col - 0.5)
    ax.set_ylim(-0.5, n_row - 0.5)
    ax.set_aspect("equal")

    square_length = (
        ax.transData.transform([1, 0])[0] - ax.transData.transform([0, 0])[0]
    )
    marker_size = (square_length * 72 / dpi * 0.8) ** 2
    g = sns.scatterplot(
        x="x",
        y="y",
        hue="val",
        size="size",
        hue_norm=(-0.5, 0.5),
        palette=cmap,
        marker="s",
        sizes=(0, marker_size),
        linewidth=0,
        legend=False,
        data=df_plot,
        ax=ax,
    )
    plt.draw()
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [t.get_text() for t in ax.get_xticklabels()],
        rotation=xticklabels_rotation,
        va="top",
        ha="right",
        fontsize=10,
    )
    ax.tick_params(left=False, bottom=False, pad=-2)
    trans = mtrans.Affine2D().translate(5, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax