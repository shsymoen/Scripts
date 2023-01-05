#!/usr/bin/env python3


def scatter_plot_color(
    f,
    ax,
    df,
    xas,
    yas,
    colorcat,
    markersize=3,
    colormap="viridis",
    downsample=True,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing

    if downsample:
        while len(df[xas]) > 5000:
            df = df.iloc[::5]

    x = df[xas]
    y = df[yas]

    if is_object(df[colorcat]):
        le = preprocessing.LabelEncoder()
        colors = le.fit_transform(df[colorcat])
        n = len(df[colorcat].unique())
        sct = ax.scatter(
            x=x, y=y, c=colors, s=markersize, cmap=plt.cm.get_cmap(colormap, n)
        )
        # This function formatter replaces the ticks with target names
        formatter = plt.FuncFormatter(
            lambda val, loc: le.inverse_transform([val])[0]
        )
        # We must make sure the ticks match our target names
        cbar = f.colorbar(sct, ticks=colors, format=formatter)
        # Set the clim so that labels are centered on each block
        sct.set_clim(-0.5, n - 0.5)
    elif is_datetime(df[colorcat]):
        colors = df[colorcat]
        sct = ax.scatter(x=x, y=y, c=colors, s=markersize, cmap=colormap)
        cbar = f.colorbar(sct, ax=ax, orientation="vertical")
        cbar.ax.set_yticklabels(
            pd.to_datetime(cbar.get_ticks()).strftime(date_format="%b %Y")
        )
    else:
        colors = df[colorcat]
        sct = ax.scatter(x=x, y=y, c=colors, s=markersize, cmap=colormap)
        cbar = f.colorbar(sct)

    ax.set(
        xlabel=xas,
        ylabel=yas,
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


def is_datetime(array_like):
    """is_datetime.

    Parameters
    ----------
    array_like :
        DataFrame
    """
    return array_like.dtype.name == "datetime64[ns]"


def xlim_to_01(lowerlimit, upperlimit, percentage):
    """Calculates the position for a text element"""
    return lowerlimit + percentage / 100 * (upperlimit - lowerlimit)


def xpos_to_ypos(xpos, xp, fp):
    """Calculates the linear interpolated value"""
    import numpy as np

    return np.interp(xpos, xp, fp)


def f_save(f, f_name, dpi=300):
    """Saves a matplotlib figure to a given location

    Parameters
    ----------
    f : matplotlib.figure.Figure object
    f_name : string
        Name of the figure to be created.
        The string can include the folder location.
    dpi : int, optional (default: 300)
        Dots per inch

    Returns
    -------
    None
    """
    f.savefig(
        f_name + ".png",
        format="png",
        dpi=dpi,
    )
    return None


def add_intervals_parity_plot(
    ax,
):
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


def create_parity_plot(
    ax,
    df,
    reference_col_name,
    parityplot_col_names,
    uom,
    add_interval_lines=True,
    add_r2_score=True,
):
    """Creates a parity plot for a specific
    component for different simulations
    """

    no = len(parityplot_col_names)
    colors, markers = colors_markers(no)
    single = False

    # Check if single comparison or multiple
    if not isinstance(parityplot_col_names, list):
        parityplot_col_names = [parityplot_col_names]
        single = True

    # Iterate over all different comparison columns in the DataFrame
    for color, marker, parityplot_col_name in zip(
        colors, markers, parityplot_col_names
    ):
        lbl_legend = parityplot_col_name
        if add_r2_score:
            from sklearn.metrics import r2_score

            r2 = r2_score(df[reference_col_name], df[parityplot_col_name])
            lbl_legend = "{} - $R^2$: {:.2%}".format(lbl_legend, r2)

        fc = color if single else "white"
        ax.scatter(
            x=df[reference_col_name],
            y=df[parityplot_col_name],
            marker=marker,
            s=20,
            facecolor=fc,
            edgecolor=color,
            label=lbl_legend,
        )

    # Add interval lines
    if add_interval_lines:
        add_intervals_parity_plot(ax)

    # Set title and labels
    if not single:
        ylabel = "{}".format(uom)
    else:
        ylabel = "{}, {}".format(parityplot_col_names[0], uom)
    ax.set(
        xlabel="{}, {}".format(reference_col_name, uom),
        ylabel=ylabel,
    )
    if not single:
        handles, labels = ax.get_legend_handles_labels()
        number = len(parityplot_col_names)
        ax.legend(
            handles=handles[:number],
            labels=labels[:number],
            loc="best",
            scatterpoints=1,
            fontsize=10,
            frameon=False,
        )


def colors_markers(no):
    """colors_markers. Provides a list of markers and colors for matplotlib
    graphs with a length of number 'no'

    Parameters
    ----------
    no :
        int, number of markers and colors required for the figure
    Returns
    -------
    markers :
        lst, list with markers of size no
    colors :
        lst, list with colors of size no
    """

    markers = ["o", "^", "s", "D", "*", "<", ">", "v", "p", "d", "H", "8", "h"]
    colors = [
        "g",
        "darkblue",
        "r",
        "deepskyblue",
        "darkmagenta",
        "coral",
        "indianred",
    ]

    return colors[:no], markers[:no]


def create_PCA_figure(
    ax,
    pca_results,
    pca_object,
    colors,
    add_title=True,
    pcs=[1, 2],
    loading=False,
    loading_labels=None,
):
    """Creates a score plot with the first 2 PC's of the PCA
    together with the color_encoding as it is given in color_encoding

    Parameters
    ----------
    ax : Matplotlib axis object
    pca_results:
    expl_variance :
    color_encoding : String
    ax : Matplotlib axis object

    Returns
    -------
    Score plot PC1 and PC2
    """
    import pandas as pd

    expl_variance = pca_object.explained_variance_ratio_
    print("Explained variance:", pd.Series(expl_variance[:5]))
    x = pcs[0] - 1
    y = pcs[1] - 1
    img = ax.scatter(
        pca_results[:, x],
        pca_results[:, y],
        c=colors,
    )

    ax.set(
        xlabel="PC {} ({:.2f})".format(pcs[0], expl_variance[x]),
        ylabel="PC {} ({:.2f})".format(pcs[1], expl_variance[y]),
    )
    if add_title:
        ax.set(
            title="PCA",
        )

    if loading:
        loading_plotter(ax, pca_object, loading_labels)

    return img


def pca_processor(df, scaler="MinMax"):
    """pca_processor.

    Parameters
    ----------
    df :
        DataFrame for which the Principal Component analysis will be performed
    scaler :
        string/None to indicate the preprocessing done on the DataFrame
        (default: MinMax)
    """
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.decomposition import PCA  # Principal component analysis

    # Drop rows that contain infinite or NaN data as PCA cannot process these
    # datapoints
    with pd.option_context("mode.use_inf_as_null", True):
        df_pca = df.dropna(how="any")

    # Perform preprocessing for PCA. Either MinMax or StandardScaler from the
    # sklearn library
    if scaler == "MinMax":
        scaler = preprocessing.MinMaxScaler()
        # Scale the data according to the selected scaler
        df_pca_scaled = scaler.fit_transform(df_pca.values)
    elif scaler == "Standard":
        scaler = preprocessing.StandardScaler()
        # Scale the data according to the selected scaler
        df_pca_scaled = scaler.fit_transform(df_pca.values)
    elif scaler is None:
        df_pca_scaled = df_pca
    else:
        print("ERROR: No valid scaler selected. Chose: MinMax, or Standard")

    # Perform the Principal Component Analysis
    pca = PCA()
    pca_results = pca.fit_transform(df_pca_scaled)

    return pca, pca_results


def loading_plotter(ax, pca_object, labels=None):
    import numpy as np
    import pandas as pd

    # Make own labels if no names are given as input
    if labels is None:
        labels = ["Var" + str(i + 1) for i in range(n)]
    loading_coeff_all = np.transpose(pca_object.components_[0:2, :])
    # Create a DataFrame from the loadings
    df_loadings_all = pd.DataFrame(
        loading_coeff_all, index=labels, columns=["x", "y"]
    )
    # Add the magnitude of the loading vectors to the DataFrame for sorting
    # and take the 5 most import contributions
    df_loadings_all["norm"] = [
        np.linalg.norm(vector) for vector in loading_coeff_all
    ]
    df_loading = df_loadings_all.sort_values(by="norm", ascending=False).iloc[
        :5
    ]
    alpha = 0.3

    loading_coeff = df_loading[["x", "y"]].values
    labels = df_loading.index
    n = loading_coeff.shape[0]
    for i in range(n):
        ax.arrow(
            0,
            0,
            loading_coeff[i, 0],
            loading_coeff[i, 1],
            color="r",
            alpha=0.5,
        )
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat")
        ax.text(
            loading_coeff[i, 0] * 1.15,
            loading_coeff[i, 1] * 1.15,
            labels[i],
            color="g",
            ha="center",
            va="center",
            bbox=props,
        )

    return ax


def create_tsne_figure(
    ax,
    tsne_results,
    colors,
    add_title=True,
):
    """Creates a t-SNE embedding
    of the columns that are given in the column names,
    together with the color_encoding as it is given in color_encoding

    Parameters
    ----------
    ax : Matplotlib axis object
    color_encoding : String

    Returns
    -------
    t-SNE embedding
    """
    img = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors)

    if add_title:
        ax.set(title="t-SNE")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    return img


def create_widgets_interactive(df):
    import pandas as pd

    # For the widgets in the interactive display
    from ipywidgets import (
        BoundedIntText,
        Box,
        Button,
        Checkbox,
        Dropdown,
        FloatRangeSlider,
        FloatText,
        HBox,
        IntSlider,
        Label,
        Output,
        SelectionRangeSlider,
        Tab,
        Text,
        ToggleButton,
        VBox,
    )

    # Create all widgets
    # widget for filtering the DataFrame
    sliders = {}
    for col in df.columns:
        if df[col].dtype == "float":
            minn = df[col].min()
            maxx = df[col].max()
            stepsize = abs(maxx - minn) / 1000
            range_slider = FloatRangeSlider(
                value=[minn, maxx],
                min=minn,
                max=maxx,
                step=stepsize,
                description=col,
                readout_format=".2g",
            )
            sliders[col] = range_slider
        elif pd.api.types.is_datetime64_dtype(df[col]):
            minn = df[col].min()
            maxx = df[col].max()
            fmt = "%Y-%m-%d"
            date_range = pd.date_range(start=minn, end=maxx, freq="D")
            options = [(item.strftime(fmt), item) for item in date_range]
            range_slider = SelectionRangeSlider(
                options=options, index=(0, len(options) - 1)
            )
            sliders[col] = range_slider

    # Wich columns to plot
    xas_widget = Dropdown(
        options=list(df.columns), description="x-axis", value=df.columns[0]
    )
    yas_widget = Dropdown(
        options=list(df.columns), description="y-axis", value=df.columns[1]
    )
    color_widget = Dropdown(
        options=list(df.columns), description="coloring", value=df.columns[1]
    )

    plot_button = Button(
        description="Plot",
    )

    save_button = Button(
        description="Save figure",
    )

    figure_name = Text(
        value="figure_name", placeholder="Type something", disabled=False
    )
    figure_title = Text(
        value="figure_title", placeholder="Type something", disabled=False
    )

    grid_button = Checkbox(value=False, description="Grid")

    add_interval_button = Checkbox(
        value=False,
        description="intervals",
    )
    marker_size_input = BoundedIntText(
        value=20,
        min=1,
        max=50,
        step=5,
        description="Marker size",
        disabled=False,
    )

    if df[xas_widget.value].dtype == "float":
        xrange = df[xas_widget.value].max() - df[xas_widget.value].min()
        xlim_min = round(df[xas_widget.value].min() - 0.1 * xrange, 2)
        xlim_max = round(df[xas_widget.value].max() + 0.1 * xrange, 2)
    else:
        xlim_min, xlim_max = 0, 100
    if df[yas_widget.value].dtype == "float":
        yrange = df[yas_widget.value].max() - df[yas_widget.value].min()
        ylim_min = round(df[yas_widget.value].min() - 0.1 * yrange, 2)
        ylim_max = round(df[yas_widget.value].max() + 0.1 * yrange, 2)
    else:
        ylim_min, ylim_max = 0, 100

    xlim_min_widget = FloatText(
        value=xlim_min,
        step=0.1,
        description="x-limit",
    )
    xlim_max_widget = FloatText(
        value=xlim_max,
        step=0.1,
        description="- ",
    )
    ylim_min_widget = FloatText(
        value=ylim_min,
        step=0.1,
        description="y-limit",
    )
    ylim_max_widget = FloatText(
        value=ylim_max,
        step=0.1,
        description="- ",
    )

    def on_value_change_xas_widget(change):
        if df[change["new"]].dtype == "float":
            xrange = df[change["new"]].max() - df[change["new"]].min()
            xlim_min_widget.value = round(
                df[change["new"]].min() - 0.1 * xrange, 2
            )
            xlim_max_widget.value = round(
                df[change["new"]].max() + 0.1 * xrange, 2
            )
        # No axes limits can be set (yet) when the selected column is a
        # datetime object
        else:
            pass

    def on_value_change_yas_widget(change):
        if df[change["new"]].dtype == "float":
            yrange = df[change["new"]].max() - df[change["new"]].min()
            ylim_min_widget.value = round(
                df[change["new"]].min() - 0.1 * yrange, 2
            )
            ylim_max_widget.value = round(
                df[change["new"]].max() + 0.1 * yrange, 2
            )
        # No axes limits can be set (yet) when the selected column is a
        # datetime object
        else:
            pass

    xas_widget.observe(on_value_change_xas_widget, names="value")
    yas_widget.observe(on_value_change_yas_widget, names="value")

    # Create the tabs to interact
    # with the widgets
    sliderbox = [
        HBox(children=[Label(sliders[slider].description), sliders[slider]])
        for slider in sliders
    ]

    tab2 = VBox(children=sliderbox)
    tab1 = HBox(
        children=[
            VBox(children=[xas_widget, yas_widget, color_widget]),
            VBox(
                children=[
                    HBox(children=[grid_button, add_interval_button]),
                    marker_size_input,
                    HBox(children=[xlim_min_widget, xlim_max_widget]),
                    HBox(children=[ylim_min_widget, ylim_max_widget]),
                ]
            ),
        ]
    )

    tab = Tab(children=[tab1, tab2])
    tab.set_title(0, "plot")
    tab.set_title(1, "filtering")

    return (
        sliders,
        xas_widget,
        yas_widget,
        color_widget,
        plot_button,
        save_button,
        figure_name,
        grid_button,
        add_interval_button,
        marker_size_input,
        figure_title,
        xlim_min_widget,
        xlim_max_widget,
        ylim_min_widget,
        ylim_max_widget,
        tab,
    )
