import matplotlib.colors as mc
import colorsys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.transforms as mtrans
from typing import Dict, List
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


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


def plot_calibration(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    predstd_col: str,
    group_col=None,
    ax=None,
    jitter=0.3,
    n=10,
    random_state=1,
):
    if ax is None:
        ax = plt.gca()

    if group_col is not None:
        df_grouped = df.groupby(group_col)
        group_labels = df_grouped.groups
        n_group = len(group_labels)
    else:
        df_grouped = [("all", df)]
        group_labels = ["All"]
        n_group = 1

    for i, (group, df_group) in enumerate(df_grouped):
        df_group = df_group.sample(n=n, random_state=random_state)

        x = i + np.linspace(-0.5, 0.5, len(df_group)) * jitter
        ymean = df_group[pred_col]
        yerr = df_group[predstd_col]

        eb = ax.errorbar(
            x=x, y=ymean, yerr=yerr, fmt="none", capsize=0, lw=1.0, color="gray"
        )
        # eb[-1][0].set_linestyle("--")
        ax.scatter(x=x, y=df_group[y_col], s=4, color="red", zorder=10)

    # xlabel
    ax.set_xlim(-0.5, n_group - 0.5)
    ax.set_xticks(np.arange(n_group))
    ax.set_xticklabels(group_labels)
    if n_group > 1:
        ax.set_xlabel(group_col)


def compare_values(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = None,
    ylabel: str = None,
    ax=None,
    s: int = 5,
):
    """Compare two p-values.
    Parameters
    ----------
    x_pval: np.ndarray
        The p-value for the first variable.
    y_pval: np.ndarray
        The p-value for the second variable.
    xlabel: str
        The label for the first variable.
    ylabel: str
        The label for the second variable.
    ax: matplotlib.Axes
        A matplotlib axes object to plot on. If None, will create a new one.
    """
    if ax is None:
        ax = plt.gca()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    nonnan_idx = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[nonnan_idx], y[nonnan_idx]
    ax.scatter(x, y, s=s)
    lim = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) * 1.1
    ax.axline((0, 0), slope=1, color="k", ls="--", alpha=0.5, lw=1, label="y=x")

    # add a regression line
    slope = np.linalg.lstsq(x[:, None], y[:, None], rcond=None)[0].item()

    ax.axline(
        (0, 0),
        slope=slope,
        color="black",
        ls="--",
        lw=1,
        label=f"y={slope:.2f} x",
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.legend()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_heatmap(
    df_val: pd.DataFrame,
    df_annot: pd.DataFrame = None,
    annot_kws: Dict = None,
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
):
    """
    Plot heatmap with annotations.
    df_val: pd.DataFrame
        The dataframe with the values to plot.
    df_annot: pd.DataFrame
        The dataframe with the annotations to plot.
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


def plot_r2_heatmap(
    df_val: pd.DataFrame,
    df_annot: pd.DataFrame,
    cbar_pad=0.04,
    cbar_fraction=0.0188,
    squaresize=45,
    heatmap_vmin=-0.5,
    heatmap_vmax=0.5,
    heatmap_linecolor="white",
    heatmap_linewidths=1.0,
    dpi=150,
):
    """Plot heatmap of variable R2

    Parameters
    ----------
    df_val : pd.DataFrame
        R2 differences.
    df_annot : pd.DataFrame
        annotations.
    cbar_pad : float, optional
        pad for colorbar, by default 0.04
    cbar_fraction : float, optional
        fraction for colorbar, by default 0.0188

    Returns
    -------
    fig, ax
    """
    fig, ax = plot_heatmap(
        df_val=df_val,
        df_annot=df_annot,
        annot_kws={"fontsize": 6, "weight": "bold"},
        cmap=plt.get_cmap("bwr", 11),
        squaresize=squaresize,
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax,
        heatmap_linecolor=heatmap_linecolor,
        heatmap_linewidths=heatmap_linewidths,
        heatmap_cbar_kws=dict(
            use_gridspec=False,
            location="right",
            fraction=cbar_fraction,
            pad=cbar_pad,
            drawedges=True,
        ),
        dpi=dpi,
    )
    ax.set_xlabel(None)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)

    ax.set_ylabel(None)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_ticklabels(["<-50%", "-25%", "0%", "25%", ">50%"])
    cbar.ax.set_ylabel(
        "Relative $\Delta (R^2)$", rotation=270, fontsize=9, labelpad=6.0
    )
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)

    cbar.ax.tick_params(labelsize=8)
    cbar.ax.tick_params(size=0)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    return fig, ax


def _group_plot(
    df,
    val_col,
    groups,
    axes,
    pos_offset,
    color,
    plot_type="box",
):
    """Box / line plots for each group (in each panel)
    df should contain "group", "subgroup"
    each group corresponds to a panel, each subgroup corresponds to
    different x within the panel

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing 'group', 'subgroup', val_col
    val_col : str
        column containing the values
    """
    assert plot_type in ["box", "line"]

    for group_i, group in enumerate(groups):
        df_group = df[df.group == group]
        dict_val = {
            group: df_tmp[val_col].values
            for group, df_tmp in df_group.groupby("subgroup")
        }
        x = list(dict_val.keys())
        vals = list(dict_val.values())
        means = [np.mean(_) for _ in vals]
        sems = [np.std(_) / np.sqrt(len(_)) for _ in vals]
        if plot_type == "box":
            props = {"linewidth": 0.75}
            bplot = axes[group_i].boxplot(
                positions=np.arange(len(vals)) + pos_offset,
                x=vals,
                sym="",
                widths=0.15,
                patch_artist=True,
                boxprops=props,
                whiskerprops=props,
                capprops=props,
                medianprops=props,
            )
            for patch in bplot["boxes"]:
                patch.set_facecolor(color)
            for patch in bplot["medians"]:
                patch.set_color("black")

        elif plot_type == "line":
            axes[group_i].errorbar(
                x=np.arange(len(vals)) + 1 + pos_offset,
                y=means,
                yerr=sems,
                fmt=".--",
                ms=4,
                mew=1,
                linewidth=1,
                color=color,
            )
        else:
            raise ValueError("plot_type must be 'box' or 'line'")

        axes[group_i].set_xlabel(group)
        axes[group_i].set_xticks(np.arange(len(vals)))
        axes[group_i].set_xticklabels(x)


def plot_group_r2(df: pd.DataFrame, figsize=(7, 1.5), groups=None, width_ratios=None):
    """Plot R2 by groups

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing 'group', 'subgroup', 'r2'
    figsize : tuple, optional
        figure size, by default (7, 1.5)
    """

    if groups is None:
        groups = df["group"].unique()
    if width_ratios is None:
        width_ratios = (
            np.array([len(df[df["group"] == g]["subgroup"].unique()) for g in groups])
            + 3
        )

    fig, axes = plt.subplots(
        figsize=figsize,
        dpi=150,
        ncols=len(width_ratios),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )

    for i, group in enumerate(groups):
        r2 = df[df["group"] == group].groupby("subgroup").mean()["r2"].values
        r2_se = df[df["group"] == group].groupby("subgroup").sem()["r2"].values

        axes[i].bar(
            x=np.arange(len(r2)),
            height=r2,
            yerr=r2_se * 2,
            edgecolor="k",
            linewidth=1,
            alpha=0.6,
            color="#FFA500",
            width=0.6,
        )
        axes[i].set_xlim(-1, len(r2))
        axes[i].set_xticks(np.arange(len(r2)))
        axes[i].set_xlabel(group)
    axes[0].set_ylabel("$R^2 (y, \widehat{y})$", fontsize=12)
    return fig, axes


def plot_group_predint(
    df: pd.DataFrame,
    figsize=(7, 1.8),
    methods: List = None,
    method_colors: Dict = None,
    groups=None,
    pos_offset: float = 0.2,
    legend_bbox_to_anchor=(0.5, 0.96),
    width_ratios=None,
):
    """Plot the prediction interval summary

    Parameters
    ----------
    df : pd.DataFrame
        df contains: 'method', 'group', 'subgroup', 'coverage', 'length'

    pos_offset : float, optional
        position offset
    """
    # plot 2 figures
    if methods is None:
        methods = df["method"].unique()
    n_method = len(methods)
    if method_colors is None:
        palatte = sns.color_palette("Set1", n_method)
        method_colors = {method: color for method, color in zip(methods, palatte)}

    assert len(method_colors) == n_method
    if groups is None:
        groups = df["group"].unique()
    if width_ratios is None:
        width_ratios = (
            np.array([len(df[df["group"] == g]["subgroup"].unique()) for g in groups])
            + 3
        )

    fig_list = []
    axes_list = []
    for val_col in ["coverage", "length"]:
        fig, axes = plt.subplots(
            figsize=figsize,
            ncols=len(width_ratios),
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios},
            dpi=150,
        )

        for i, method in enumerate(methods):
            _group_plot(
                df[df["method"] == method],
                val_col=val_col,
                groups=groups,
                pos_offset=-pos_offset * (len(methods) - 1) / 2 + pos_offset * i,
                axes=axes,
                color=method_colors[method],
            )
        legend_elements = [
            Patch(facecolor=method_colors[method], edgecolor="k", label=method)
            for method in methods
        ]
        fig.legend(
            handles=legend_elements,
            loc="center",
            ncol=len(methods),
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=8,
            frameon=False,
        )
        if val_col == "coverage":
            axes[0].set_ylabel("Coverage")
            axes[0].yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.0%}".format(y))
            )

        elif val_col == "length":
            axes[0].set_ylabel("Length")

        fig_list.append(fig)
        axes_list.append(axes)
    return fig_list[0], axes_list[0], fig_list[1], axes_list[1]


def plot_r2_cov(
    dict_r2: Dict,
    dict_cov1: Dict,
    dict_cov2: Dict,
    trait: str,
    xlabels: List,
    xlabel_map: Dict,
    ylim_r2: List = None,
    ylim_cov: List = None,
    figsize=None,
):
    """
    Plot the figure for both R2 and coverages as a function of covariates.

    Parameters
    ----------
    dict_r2: Dict [group -> pd.DataFrame]
        for each group [n_group, 2] / [n_group, 1]
        [n_group, 2]: 1st column: R2, 2nd column: standard error
        [n_group, 1]: 1st column: R2
        similarly for dict_cov1 and dict_cov2
    dict_cov1: Dict[group -> np.ndarray]
        each group corresponds to a coverage matrix measuring overall calibration
    dict_cov2: Dict[group -> np.ndarray]
        each group corresponds to a coverage matrix measureing by-covariate calibration
    trait: str
        trait name, used for labels.
    """
    width_ratios = [dict_r2[x].shape[0] for x in xlabels]
    if figsize is None:
        figsize = (0.3 * sum(width_ratios) + 0.1 * len(width_ratios), 3.5)
        print("Default figsize:", figsize)
    fig, axes = plt.subplots(
        figsize=figsize,
        dpi=150,
        nrows=2,
        ncols=len(xlabels),
        sharex="col",
        sharey="row",
        gridspec_kw={"width_ratios": width_ratios},
    )
    for i, group in enumerate(xlabels):
        ax = axes[0, i]
        df_r2 = dict_r2[group]
        n_group = df_r2.shape[0]
        if df_r2.shape[1] == 2:
            mean, sd = df_r2.iloc[:, 0], df_r2.iloc[:, 1]
        else:
            mean = df_r2.iloc[:, 0]
            sd = np.zeros(n_group)

        ax.errorbar(
            x=np.arange(n_group),
            y=mean,
            yerr=sd,
            fmt=".--",
            ms=4,
            mew=1,
            linewidth=1,
            color="black",
        )
        ax.set_xlim(-0.5, n_group - 0.5)
        if ylim_r2 is not None:
            ax.set_ylim(*ylim_r2)
        if i == 0:
            ax.set_ylabel(f"{trait} $R^2$")

    # coverage
    for i, group in enumerate(xlabels):
        ax = axes[1, i]

        ax.axhline(y=0.9, ls="--", lw=1.0, color="black", alpha=0.5)
        if ylim_cov is not None:
            ax.set_ylim(*ylim_cov)
        if i == 0:
            ax.set_ylabel(f"{trait} coverage")

        for cov_i, dict_cov in enumerate([dict_cov1, dict_cov2]):
            cov_vals = dict_cov[group].values
            if cov_vals.shape[1] == 2:
                cov_mean, cov_sd = cov_vals[:, 0], cov_vals[:, 1]
            else:
                cov_mean = cov_vals[:, 0]
                cov_sd = np.zeros(cov_mean.shape)
            n_group = cov_vals.shape[0]
            label = ["Overall", "By-cov"][cov_i]
            ax.errorbar(
                x=np.arange(n_group) - 0.1 + cov_i * 0.2,
                y=cov_mean,
                yerr=cov_sd,
                fmt=".--",
                ms=4,
                mew=1,
                linewidth=0.8,
                label=label,
            )
        ax.set_xticks(np.arange(n_group))
        ax.set_xlabel(xlabel_map[group])

    return fig, axes