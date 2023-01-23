#!/usr/bin/env python3


def pca_processor(df_from_which_pca, df_to_add_pca, scaler="MinMax"):
    """pca_processor.

    Parameters
    ----------
    df_from_which_pca :
        DataFrame for which the Principal Component analysis will be performed
    df_to_add_pca :
        DataFrame on which the Principal Component analysis results are added 
        Note that df_from_which_pca and df_to_add_pca need to have the same
        length
    scaler :
        string/None to indicate the preprocessing done on the DataFrame
        (default: MinMax)

    Returns
    -------
    pca : PCA sklearn object
    pca_results : data transformed in PCs
    df_and_pca : DataFrame
        original DataFrame with the PCs added to it

    """
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.decomposition import PCA  # Principal component analysis

    if len(df_from_which_pca.index) != len(df_to_add_pca.index):
        print(
            "df_from_which_pca and df_to_add_pca do not have the "
            "same length hence the function cannot be executed to "
            "create a new DataFrame."
        )
        return

    df_from_which_pca = df_from_which_pca.reset_index(drop=True)
    df_to_add_pca = df_to_add_pca.reset_index(drop=True)

    # Drop rows that contain infinite or NaN data as PCA cannot process these
    # datapoints
    with pd.option_context("mode.use_inf_as_null", True):
        df_from_which_pca = df_from_which_pca.dropna(how="any")
    # Make sure that the two DataFrames stay the same length even if some
    # infinite or NaN data is dropped in previous line of code
    df_to_add_pca = df_to_add_pca.loc[df_from_which_pca.index]

    # Add preprocessing for categorical data
    ### still to be added ###

    # Perform preprocessing for PCA. Either MinMax or StandardScaler from the
    # sklearn library
    if scaler == "MinMax":
        scaler = preprocessing.MinMaxScaler()
        # Scale the data according to the selected scaler
        df_pca_scaled = scaler.fit_transform(df_from_which_pca.values)
    elif scaler == "Standard":
        scaler = preprocessing.StandardScaler()
        # Scale the data according to the selected scaler
        df_pca_scaled = scaler.fit_transform(df_from_which_pca.values)
    elif scaler is None:
        df_pca_scaled = df_from_which_pca
    else:
        print(
            "ERROR: No valid scaler selected. Chose: MinMax, Standard or None"
        )

    # Perform the Principal Component Analysis
    pca = PCA()
    pca_results = pca.fit_transform(df_pca_scaled)
    pc_col_names = [
        "PC {} ({:.2%})".format(i + 1, var)
        for i, var in enumerate(pca.explained_variance_ratio_)
    ]
    pca_results_df = pd.DataFrame(data=pca_results, columns=pc_col_names,)
    df_and_pca = pd.concat([df_to_add_pca, pca_results_df], axis=1)

    return pca, pca_results, df_and_pca
