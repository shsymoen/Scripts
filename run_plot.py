#!/usr/bin/env python3

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import datasets

    from plotter import scatter_plot_color

    iris = datasets.load_iris()
    df_iris = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    converter = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df_iris["target"] = df_iris["target"].replace(converter)
    print(df_iris.head())

    f, ax = plt.subplots()
    scatter_plot_color(
        f, ax, df_iris, "sepal width (cm)", "sepal length (cm)", "target"
    )

    f.tight_layout()
    f.show()

    input()
