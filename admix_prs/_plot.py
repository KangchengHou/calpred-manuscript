import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np


def plot_calibration(
    df: pd.DataFrame,
    y_col: str,
    lower_col: str,
    upper_col: str,
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
        group_labels = "All"
        n_group = 1

    for i, (group, df_group) in enumerate(df_grouped):
        df_group = df_group.sample(n=n, random_state=random_state)

        x = i + np.linspace(-0.5, 0.5, len(df_group)) * jitter
        ymean = (df_group[upper_col] + df_group[lower_col]) / 2
        yerr = (df_group[upper_col] - df_group[lower_col]) / 2

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
