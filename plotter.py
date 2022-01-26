#!/usr/bin/env python3


def scatter_plot_color(
    f, ax, df, xas, yas, colorcat, colormap="viridis",
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing

    x = df[xas]
    y = df[yas]

    if is_object(df[colorcat]):
        le = preprocessing.LabelEncoder()
        colors = le.fit_transform(df[colorcat])
        n = len(df[colorcat].unique())
        sct = ax.scatter(x=x, y=y, c=colors, cmap=plt.cm.get_cmap(colormap, n))
        # This function formatter replaces the ticks with target names
        formatter = plt.FuncFormatter(
            lambda val, loc: le.inverse_transform([val])[0]
        )
        # We must make sure the ticks match our target names
        cbar = f.colorbar(sct, ticks=colors, format=formatter)
        # Set the clim so that labels are centered on each block
        sct.set_clim(-0.5, n - 0.5)
    else:
        colors = df[colorcat]
        sct = ax.scatter(x=x, y=y, c=colors, cmap=colormap)
        cbar = f.colorbar(sct)

    ax.set(
        xlabel=xas, ylabel=yas,
    )
    cbar.ax.set_ylabel(colorcat)

    return f, ax, sct, cbar


def lin_func_plotter(f, ax, df, xas, yas, add_text=True):
    import numpy as np

    x = df[xas]
    y = df[yas]
    idx = np.isfinite(x) & np.isfinite(y)
    d = np.polyfit(x[idx], y[idx], 1)
    func = np.poly1d(d)

    ax.plot(x, func(x), color="grey", ls=":")
    if add_text:
        ax.text(
            10,
            func(10) + 18,
            "{}".format(func),
            size=8,
            va="center",
            ha="center",
        )
    return f, ax, func


def is_object(array_like):
    """is_object.

    Parameters
    ----------
    array_like :
        DataFrame
    """
    return array_like.dtype.name == "object"


def xlim_to_01(lowerlimit, upperlimit, percentage):
    """ Calculates the position for a text element """
    return lowerlimit + percentage / 100 * (upperlimit - lowerlimit)


def xpos_to_ypos(xpos, xp, fp):
    """ Calculates the linear interpolated value """
    import numpy as np

    return np.interp(xpos, xp, fp)


def add_intervals_parity_plot(ax,):
    """add_intervals_parity_plot.

    Parameters
    ----------
    ax :
        matplotlib.axes.Axes
    """
    # Get limits of plot
    if ax.get_xlim()[0] < ax.get_ylim()[0]:
        lowest = ax.get_xlim()[0]
    else:
        lowest = ax.get_ylim()[0]
    if lowest < 0:
        lowest = 0
    if ax.get_xlim()[1] > ax.get_ylim()[1]:
        highest = ax.get_xlim()[1]
    else:
        highest = ax.get_ylim()[1]

    # Make a square plot
    ax.set_xlim((lowest, highest))
    ax.set_ylim((lowest, highest))
    # Define some formatting
    bbox_props = dict(facecolor="white", edgecolor="none", pad=1, alpha=0.5)
    # Get ax limits
    xlow = ax.get_xlim()[0]
    xhigh = ax.get_xlim()[1]
    ylow = ax.get_ylim()[0]
    yhigh = ax.get_ylim()[1]
    xp = [xlow, xhigh]
    yp = [ylow, yhigh]

    # Put middle line
    ax.plot((0, highest), (0, highest), c=".3", alpha=0.5)

    # Put 5, 10, and 20% ranges
    line_dct = {
        95: {"ls": "dotted", "c": ".2", "perc_name": ["-5%", "+5%"]},
        90: {"ls": "-.", "c": ".4", "perc_name": ["-10%", "+10%"]},
        80: {"ls": "--", "c": ".6", "perc_name": ["-20%", "+20%"]},
    }
    for percentage in line_dct:
        ax.plot(
            (0, highest * percentage / 100),
            (0, highest),
            ls=line_dct[percentage]["ls"],
            c=line_dct[percentage]["c"],
            alpha=0.5,
        )
        ax.plot(
            (0, highest),
            (0, highest * percentage / 100),
            ls=line_dct[percentage]["ls"],
            c=line_dct[percentage]["c"],
            alpha=0.5,
        )
        # Add percentage indications to the lines
        xposfix = xlim_to_01(xlow, xhigh, percentage)
        yposfix = xlim_to_01(ylow, yhigh, percentage)
        # Negative percentage
        ax.text(
            xposfix,
            xpos_to_ypos(xposfix, xp, [ylow, highest * percentage / 100]),
            line_dct[percentage]["perc_name"][0],
            alpha=0.5,
            va="center",
            ha="center",
            bbox=bbox_props,
        )
        # Positve percentage
        ax.text(
            xpos_to_ypos(yposfix, yp, [xlow, highest * percentage / 100]),
            yposfix,
            line_dct[percentage]["perc_name"][1],
            alpha=0.5,
            va="center",
            ha="center",
            bbox=bbox_props,
        )
