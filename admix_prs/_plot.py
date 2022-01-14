import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


def plot_calibration(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    yerr_col: str,
    ax: matplotlib.axes.Axes = None,
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'x', 'y', 'y_pred', 'y_pred_std'
    ax : matplotlib.axes.Axes
        Axes to plot on. If None, a new figure is created.
    """
    if ax is None:
        ax = plt.gca()

    ax.errorbar(
        x=df[x_col],
        y=df[y_col],
        yerr=df[yerr_col],
        ecolor="lightgray",
        fmt=".",
        markersize=3,
        elinewidth=1.0,
        capsize=2,
    )

    ax.plot([-4, 4], [-4, 4], color="red", ls="--", alpha=0.5)
