"""data_source.py module containing functionality to extract and collate the raw data."""

import os
import glob
import json

import numpy as np

import pandas as pd


def get_subfolder_names_from_root(root_dir: str) -> list:
    """
    Retrieve a list of subfolder names from the specified root directory.

    This function lists the names of all subdirectories within the given root directory. It does
    not include files, and does not recurse into subdirectories of subdirectories.

    Parameters
    ----------
    root_dir : str
        The absolute or relative path to the root directory from which subfolder names are retrieved.

    Returns
    -------
    list of str
        A list containing the names of all subfolders directly under the specified root directory.

    Raises
    ------
    ValueError
        If the specified root directory does not exist or is not a directory.

    Examples
    --------
    Consider a directory structure as follows:
        root_dir/
        ├── subfolder1
        ├── subfolder2
        ├── .hidden
        ├── __init__.py
        └── file.txt

    >>> get_subfolder_names_from_root('root_dir')
    ['subfolder1', 'subfolder2']
    """
    if os.path.exists(root_dir):
        return [
            f
            for f in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, f)))
            and (not (f.startswith("__") or f.startswith(".")))
        ]
    else:
        raise FileNotFoundError(f"Root Directory {root_dir} does not exist.")


def get_storm_raw_data(
    root_dir: str, storm_names: list
) -> (pd.DataFrame, pd.DataFrame):
    """
    Create pandas DataFrames containing features and labels for specified storms.

    Parameters:
        - root_dir (str): The root directory where storm data is stored.
        - storm_names (list): A list of storm names.

    Returns:
        tuple: Two pandas DataFrames representing features and labels.

    Notes:
        The function searches for JSON files within the storm directories and extracts relevant data.

    Raises:
        TypeError: If the file suffix is not recognized.
    """
    features, labels = [], []
    for storm in storm_names:
        # build the storm filepath
        storm_path = os.path.join(root_dir, storm, "*.json")
        for fp in sorted(glob.glob(storm_path)):

            data = read_json(file_path=fp)

            # extract filename information
            storm_name, img_num, file_suffix = fp.split("/")[-1].split("_")
            file_data = {
                "storm_name": storm_name,
                "image_number": img_num,
                "file_suffix": file_suffix,
                "file_path": os.path.join(root_dir, storm, f"{storm}_{img_num}.jpg"),
            }

            # append file data to data
            row = {**data, **file_data}

            if file_suffix == "label.json":
                labels.append(row)

            elif file_suffix == "features.json":
                features.append(row)
            else:
                print(f"File Suffix {file_suffix} not handled.")
                pass

    df_features = pd.DataFrame(features)
    df_labels = pd.DataFrame(labels)

    return df_features, df_labels


def merge_df2_onto_df1(
    df1: pd.DataFrame, df2: pd.DataFrame, how: str, columns_to_match: list
) -> pd.DataFrame:
    """
    Merge two pandas DataFrames based on specified columns.

    This function merges `df2` onto `df1` based on the columns specified in `columns_to_match`.
    The type of merge is determined by the `how` parameter.

    Parameters
    ----------
    df1 : pd.DataFrame
        The left DataFrame to be merged.
    df2 : pd.DataFrame
        The right DataFrame to be merged.
    how : str
        The type of merge to be performed. Must be one of 'left', 'right', 'outer', 'inner'.
    columns_to_match : list of str
        The names of the columns to use as keys for the merge.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.

    Raises
    ------
    AssertionError
        If any of the specified columns in `columns_to_match` are not present in both `df1` and `df2`.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    >>> df2 = pd.DataFrame({'A': [1, 2, 4], 'C': ['x', 'y', 'z']})
    >>> merge_df2_onto_df1(df1, df2, 'inner', ['A'])
       A  B  C
    0  1  a  x
    1  2  b  y
    """

    # column check
    for _df in [df1, df2]:
        assert all(col in _df.columns for col in columns_to_match)

    df = df1.merge(
        right=df2, how=how, left_on=columns_to_match, right_on=columns_to_match
    )

    return df


def read_json(file_path):
    """
    Read a JSON file and return its content.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The content of the JSON file as a Python dictionary.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def create_time_interval_dt_column(df, time_column, dt):
    """"""
    df["dt"] = round(df[time_column].apply(pd.to_numeric) / dt).apply(int)

    return df


def get_wind_speed_data(data_dir, storm_names, numeric_cols, time_column, dt):
    """"""
    df_storm_features, df_storm_labels = get_storm_raw_data(
        root_dir=data_dir, storm_names=storm_names
    )
    df = merge_df2_onto_df1(
        df1=df_storm_features,
        df2=df_storm_labels,
        how="left",
        columns_to_match=["storm_name", "image_number"],
    )

    # create a id column (storm name + image number)
    df["id"] = df["storm_id"] + "_" + df["image_number"]
    df = df[["storm_id", "id", "relative_time", "ocean", "wind_speed"]]

    if numeric_cols is not None:
        for numeric_col in numeric_cols:
            df[numeric_col] = pd.to_numeric(df[numeric_col])

    if dt is not None:
        df = create_time_interval_dt_column(df=df, time_column=time_column, dt=dt)

    return df


def create_sequence_data(df, storm_names, input_length, target_length):
    """"""
    sequences = []
    for _storm in storm_names:
        _speed = df[df.storm_id == _storm]["wind_speed"].values
        _time = df[df.storm_id == _storm]["dt"].values
        _ocean = df[df.storm_id == _storm]["ocean"].unique()[0]
        for i in range(0, (len(_speed) - (input_length + target_length)) + 1):
            _row = np.concatenate(
                [
                    [_storm, _ocean],
                    _time[i : i + input_length + target_length],
                    _speed[i : i + input_length + target_length],
                ]
            )
            sequences.append(_row)
    columns = (
        ["storm", "ocean"]
        + [f"t_input_{i}" for i in range(input_length)]
        + [f"t_target_{i}" for i in range(target_length)]
        + [f"v_input_{i}" for i in range(input_length)]
        + [f"v_target_{i}" for i in range(target_length)]
    )

    df_sequence = pd.DataFrame(data=sequences, columns=columns)

    return df_sequence
