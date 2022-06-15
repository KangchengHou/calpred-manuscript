import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
import uncertainty_toolbox as uct
import statsmodels.api as sm
import seaborn as sns


def uct_plot(pred_mean, pred_std, y, x):

    fig, axes = plt.subplots(figsize=(15, 3), dpi=150, ncols=5)
    uct.viz.plot_xy(pred_mean, pred_std, y, x, ax=axes[0])
    axes[1].scatter(pred_mean, y, s=2, alpha=0.5)
    lim1 = min(min(pred_mean), min(y))
    lim2 = max(max(pred_mean), max(y))
    # expand limits by 10%
    lim1, lim2 = (lim1 + lim2) / 2 - (lim2 - lim1) * 0.55, (lim1 + lim2) / 2 + (
        lim2 - lim1
    ) * 0.55

    sns.regplot(x=pred_mean, y=y, ax=axes[1], scatter=False)
    plt.setp(axes[1].collections[1], alpha=0.6)

    reg_model = sm.OLS(y, sm.add_constant(pred_mean)).fit()
    intercept, slope = reg_model.params
    axes[1].axline(
        (0, intercept),
        slope=slope,
        color="black",
        ls="--",
        label=f"y={intercept:.2f}+ {slope:.2f} x",
    )
    axes[1].axline(
        (0, 0),
        slope=1,
        color="red",
        ls="--",
        label="y=x",
    )
    axes[1].set_xlim(lim1, lim2)
    axes[1].set_ylim(lim1, lim2)

    axes[1].legend()
    uct.viz.plot_intervals_ordered(pred_mean, pred_std, y, ax=axes[2])

    uct.viz.plot_calibration(pred_mean, pred_std, y, ax=axes[3])

    uct.viz.plot_adversarial_group_calibration(pred_mean, pred_std, y, ax=axes[4])
    fig.tight_layout()


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
