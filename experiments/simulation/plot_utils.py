import matplotlib.colors as mc
import colorsys
import numpy as np
import matplotlib.pyplot as plt

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

    for i,artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = lighten_color(artist.get_facecolor(), 1.2)
        artist.set_edgecolor(col)    

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*6,i*6+6):
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
        bplot = axes[group_i].boxplot(
            positions=np.arange(len(vals)) + 1 + pos_offset,
            x=vals,
            sym="",
            widths=0.15,
            patch_artist=True,
        )
        for patch in bplot["boxes"]:
            patch.set_facecolor(color)
        axes[group_i].axhline(y=0.9, color="red", lw=0.8, ls="--")
        axes[group_i].set_xlabel(group)
        axes[group_i].set_xticks(np.arange(len(vals)) + 1)
        axes[group_i].set_xticklabels(x)

    axes[0].set_ylabel("Coverage")