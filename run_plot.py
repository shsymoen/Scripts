#!/usr/bin/env python3

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import datasets

    from plotter import (
        add_intervals_parity_plot,
        create_parity_plot,
        create_PCA_figure,
        is_object,
        scatter_plot_color,
    )
    from processor import pca_processor

    iris = datasets.load_iris()
    df_iris = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    converter = {0: "setosa", 1: "versicolor", 2: "virginica"}
    #     df_iris["target"] = df_iris["target"].replace(converter)
    #     print(df_iris.head())
    #
    #     f, ax = plt.subplots()
    #     scatter_plot_color(
    #         f, ax, df_iris, "sepal width (cm)", "sepal length (cm)", "target"
    #     )
    #     f.show()
    #     input()
    #     f, ax = plt.subplots()
    #     scatter_plot_color(
    #         f,
    #         ax,
    #         df_iris,
    #         "sepal width (cm)",
    #         "sepal length (cm)",
    #         "sepal length (cm)",
    #     )
    #
    #     add_intervals_parity_plot(
    #         ax,
    #     )
    #     f.tight_layout()
    #     f.show()
    #
    #     input()
    #
    #     f, ax = plt.subplots()
    #     create_parity_plot(
    #         ax,
    #         df_iris,
    #         reference_col_name="sepal width (cm)",
    #         parityplot_col_names="sepal length (cm)",
    #         uom="cm",
    #     )
    #
    #     f.tight_layout()
    #     f.show()
    #
    #     input()
    #
    #     f, ax = plt.subplots()
    #     create_parity_plot(
    #         ax,
    #         df_iris,
    #         reference_col_name="sepal width (cm)",
    #         parityplot_col_names=["sepal length (cm)", "sepal width (cm)"],
    #         uom="cm",
    #     )
    #
    #     f.tight_layout()
    #     f.show()
    #
    #     input()

    f, ax = plt.subplots()
    pca, pca_results, df_pca = pca_processor(
        df_iris, df_iris[["sepal length (cm)", "sepal width (cm)"]]
    )
    print(df_pca.columns)

    #     scatter_plot_color(
    #         f,
    #         ax,
    #         df_pca,
    #         xas=df_pca.columns[-5],
    #         yas=df_pca.columns[-4],
    #         colorcat=df_pca.columns[1],
    #     )

    create_PCA_figure(
        f,
        ax,
        pca,
        df_pca,
        colorcat=df_pca.columns[1],
        pcs=[1, 2],
        loading=False,
        loading_labels=None,
    )

    f.tight_layout()
    f.show()

    input()
