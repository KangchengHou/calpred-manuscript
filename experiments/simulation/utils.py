import matplotlib.colors as mc
import colorsys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import calpgs
from typing import List
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def lighten_boxplot(ax):

    # https://stackoverflow.com/questions/55656683/change-seaborn-boxplot-line-rainbow-color
    def lighten_color(color, amount=0.5):
        # --------------------- SOURCE: @IanHincks ---------------------
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    for i, artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = lighten_color(artist.get_facecolor(), 1.2)
        artist.set_edgecolor(col)

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)


def group_boxplot(df, val_col, group_list=None, axes=None, pos_offset=0.0, color="C0"):
    """Box plots for each group (in each panel)
    df should contain "group", "subgroup"
    each group corresponds to a panel, each subgroup corresponds to
    different x within the panel

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing group, subgroup, `val_col`
    val_col : str
        column containing the values
    """

    if group_list is None:
        group_list = df["group"].unique()
    group_size = np.array(
        [len(df[df.group == group].subgroup.unique()) for group in group_list]
    )

    for group_i, group in enumerate(group_list):
        df_group = df[df.group == group]
        dict_val = {
            group: df_tmp[val_col].values
            for group, df_tmp in df_group.groupby("subgroup")
        }
        x = list(dict_val.keys())
        vals = list(dict_val.values())
        means = [np.mean(_) for _ in vals]
        sems = [np.std(_) / np.sqrt(len(_)) for _ in vals]
        props = {"linewidth" : 0.75}
        bplot = axes[group_i].boxplot(
            positions=np.arange(len(vals)) + 1 + pos_offset,
            x=vals,
            sym="",
            widths=0.15,
            patch_artist=True,
            boxprops=props,
            whiskerprops=props,
            capprops=props,
            medianprops=props
        )
        for patch in bplot["boxes"]:
            patch.set_facecolor(color)
            
        for patch in bplot['medians']:
            patch.set_color('black')
            
        axes[group_i].axhline(y=0.9, color="red", lw=0.8, ls="--")
        axes[group_i].set_xlabel(group)
        axes[group_i].set_xticks(np.arange(len(vals)) + 1)
        axes[group_i].set_xticklabels(x)



def simulate_data(
    df_cov: pd.DataFrame,
    var_effects: np.ndarray,
    baseline_r2: float,
    slope_effects: np.ndarray = None,
    n_train=5000,
    n_test=5000,
    n_dummy=50,
):
    """
    Simulate phenotype, PRS, with the noise level as a function of covariates

    Parameters
    ----------
    df_cov: np.ndarray
        covariates matrix
    baseline_r2: float
        R2 between PGS and phenotype for the baseline covariates
    var_effects: np.ndarray
        1d vector covariate effects
    """
    # sub-sample covariates
    df_cov = df_cov.sample(n=n_train + n_test)
    cov = df_cov.values
    n_indiv = cov.shape[0]
    if not isinstance(var_effects, np.ndarray):
        var_effects = np.array(var_effects)
    if not isinstance(slope_effects, np.ndarray):
        slope_effects = np.array(slope_effects)

    assert cov.shape[0] == n_indiv
    assert cov.shape[1] == len(var_effects)
    pred = np.random.normal(size=n_indiv)

    design = np.hstack([np.ones((n_indiv, 1)), pred.reshape(-1, 1), cov])

    true_beta = np.array([0, 1] + [0] * cov.shape[1])
    true_gamma = np.array([np.log(1 / baseline_r2 - 1), 0] + list(var_effects))
    true_slope = 1 + design @ np.concatenate([[0, 0], slope_effects])
    y = np.random.normal(
        loc=(design @ true_beta) * true_slope,
        scale=np.sqrt(np.exp(design @ true_gamma)),
    )

    df_trait = pd.concat(
        [
            pd.DataFrame({"pred": pred, "y": y}, index=df_cov.index),
            df_cov,
        ],
        axis=1,
    )
    df_trait["predstd0"] = 1.0
    for col in ["AGE", "PC1", "SEX"]:
        n_unique = len(np.unique(df_trait[col]))
        if n_unique > 5:
            col_q = pd.qcut(df_trait[col], q=5).cat.codes
        else:
            col_q = pd.Categorical(df_trait[col]).codes
        df_trait[f"{col}_q"] = col_q

    # add dummy variable
    for i in range(n_dummy):
        df_trait[f"DUMMY{i}"] = np.random.normal(size=len(df_trait))
    df_train, df_test = train_test_split(df_trait, test_size=n_test, train_size=n_train)
    return df_train, df_test


def evaluate_metrics(
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    adjust_cols: List[str],
    fit_slope: bool = False,
):
    """
    Given a dataframe to perform calibration and testing, and columns to adjust
    report (1) coverage (2) R2

    # by default, adjust_cols = ["AGE", "SEX", "PC1"]
    """

    if adjust_cols is None:
        adjust_cols = []
    df_cal, df_test = df_cal.copy(), df_test.copy()

    # interval coverage
    dict_coverage = {}
    # R2
    dict_r2 = {}
    # interval length
    dict_length = {}

    train_y = df_cal["y"]
    train_x = sm.add_constant(df_cal[["pred"]])
    train_z = sm.add_constant(df_cal[adjust_cols])

    test_x = sm.add_constant(df_test[["pred"]])
    test_z = sm.add_constant(df_test[adjust_cols])

    if fit_slope and (len(adjust_cols) > 0):
        train_slope_covar, test_slope_covar = (
            train_z.values[:, 1:],
            test_z.values[:, 1:],
        )
    else:
        train_slope_covar, test_slope_covar = None, None

    # adjust
    res = calpgs.calibrate_and_adjust(
        train_mean_covar=train_x.values,
        train_var_covar=train_z.values,
        train_y=train_y.values,
        test_mean_covar=test_x.values,
        test_var_covar=test_z.values,
        train_slope_covar=train_slope_covar,
        test_slope_covar=test_slope_covar,
    )
    df_params = pd.concat(
        [
            pd.Series(res[2], index=[col + ".beta" for col in train_x.columns]),
            pd.Series(res[3], index=[col + ".gamma" for col in train_z.columns]),
        ]
    )
    if fit_slope and (len(adjust_cols) > 0):
        df_params = pd.concat(
            [
                df_params,
                pd.Series(
                    res[4], index=[col + ".slope" for col in train_z.columns[1:]]
                ),
            ]
        )
    df_test["cal_pred"], df_test["cal_predstd"] = res[0:2]

    # evaluate
    for group_col in [None, "AGE_q", "PC1_q", "SEX_q"]:
        df_summary = calpgs.compute_group_stats(
            df_test,
            y_col="y",
            pred_col="pred",
            predstd_col="cal_predstd",
            group_col=group_col,
        )
        if group_col is None:
            dict_coverage["marginal"] = df_summary["coverage"]
            dict_r2["marginal"] = df_summary["r2"]
            dict_length["marginal"] = df_summary["length"]
        else:
            for i in df_summary.index:
                dict_coverage[f"{group_col}_{i}"] = df_summary["coverage"][i]
                dict_r2[f"{group_col}_{i}"] = df_summary["r2"][i]
                dict_length[f"{group_col}_{i}"] = df_summary["length"][i]

    return (
        pd.Series(dict_coverage),
        pd.Series(dict_r2),
        pd.Series(dict_length),
        df_params,
    )


def multi_group_plot_wrapper(
    df_stats,
    groups,
    colors,
    labels,
    ylim,
    ylabel=None,
    pos_offset=0.2,
    legend_bbox_to_anchor=(0.5, 0.96),
    val_col="coverage",
    figsize=(7, 2.2)
):
    n_group = len(groups)
    assert (len(colors) == n_group) and (len(labels) == n_group)
    df_plot = df_stats[(df_stats["adjust"].isin(groups))].copy()
    df_plot["group"] = df_plot["col"].apply(lambda x: x.split("_")[0])
    df_plot["group"] = df_plot["group"].replace(
        {"marginal": "Overall", "SEX": "Sex", "AGE": "Age"}
    )
    df_plot["subgroup"] = df_plot["col"].apply(
        lambda x: x.rsplit("_", 1)[1] if "_" in x else ""
    )

    fig, axes = plt.subplots(
        figsize=figsize,
        ncols=4,
        sharey=True,
        gridspec_kw={"width_ratios": np.array([1, 5, 2, 5]) + 3},
        dpi=150,
    )

    for i, group in enumerate(groups):
        group_boxplot(
            df_plot[df_plot["adjust"] == group],
            val_col=val_col,
            group_list=["Overall", "Age", "Sex", "PC1"],
            pos_offset=-pos_offset * (len(groups) - 1) / 2 + pos_offset * i,
            axes=axes,
            color=colors[i],
        )

    legend_elements = [
        Patch(facecolor=color, edgecolor="k", label=label)
        for color, label in zip(colors, labels)
    ]
    axes[0].set_ylim(ylim)
    axes[0].set_ylabel(ylabel)
    # Create the figure
    fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(groups),
        bbox_to_anchor=legend_bbox_to_anchor,
        fontsize=8,
        frameon=False,
    )
    return fig, axes