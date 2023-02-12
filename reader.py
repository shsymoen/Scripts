def read_pi_data_excel(excel_file_name):
    """read_pi_data_excel.

    Parameters
    ----------
    excel_file_name :
        String with name of the MS Excel file to read

    Returns
    -------
    df :
        Pandas DataFrame
    """
    import pandas as pd

    # Read in the excel file with Pandas read excel
    # The data is stored in a Pandas DataFrame called df
    df = pd.read_excel(
        excel_file_name,
        skiprows=4,  # First 4 rows are not included in the DataFrame
        header=[0, 1, 2],  # First 3 rows are seen as the header
        na_values=[
            "[-11059] No Good Data For Calculation",
            "Calc Failed",
            "No Data",
            "Bad Input",
            "Scan Off",
            "Bad",
            "Pt Created",
        ],
        parse_dates=True,  # Read the dates correctly
        thousands=".",  # Decimal is a dot
    )

    # Check if the DataFrame is loaded fully by the PI Osisoft excel plugin
    # if not fully loaded abort the code for reading the DataFrame from Excel
    for col in df.columns:
        resize = "Resize to show all values" in df[col].values
        time = "The time is invalid." in df[col].values

        if resize or time:
            raise Exception(
                "The Excel file is not fully loaded with PI data, "
                "please try again"
            )

    # join the units of measure and description of the Pi-tag into 1 column
    # name (before this command the header was a multi-index table)
    df.columns = df.columns.map(",".join)

    return df
