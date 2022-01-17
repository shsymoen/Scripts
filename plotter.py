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
