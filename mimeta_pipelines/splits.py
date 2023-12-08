import os
from typing import Callable

import pandas as pd

from mimeta_pipelines.paths import INFO_PATH


def read_split(file_name: str) -> pd.DataFrame:
    path = os.path.join(INFO_PATH, "splits", file_name)
    splits_df = pd.read_csv(path)
    return splits_df


def write_split(file_name: str, splits_df: pd.DataFrame) -> None:
    path = os.path.join(INFO_PATH, "splits", file_name)
    splits_df.to_csv(path, index=False)


def make_random_split(
    annotation_df: pd.DataFrame,
    groupby_key: list[str] | str,
    ratios: dict[str, float],
    row_filter: dict[str, list[str]] | None = None,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Creates random train/val/test splits for a dataframe

    Args:
        annotation_df (pd.DataFrame): The dataframe to split.
        groupby_key (list[str] | str): The column(s) to use to group
            data that should be in the same split.
        ratios (dict[str, float]): The ratios to split the data into.
            The dictionary may contain keys 'train', 'val', and 'test'.
        row_filter (dict[str, list[str]] | None, optional): A dictionary
            of column names and values to filter the dataframe by.
            Defaults to None.
        seed (int | None, optional): The random seed to use for
            splitting. Defaults to None.

    Returns:
        A dataframe with a column 'split' indicating which split the
        data is in.
    """
    if isinstance(groupby_key, str):
        groupby_key = [groupby_key]
    if row_filter is None:
        row_filter = {}
    # filter rows
    for col, values in row_filter.items():
        annotation_df = annotation_df[annotation_df[col].isin(values)]
    # get unique keys
    unique_groupby_keys = annotation_df[groupby_key].drop_duplicates()
    # shuffle keys
    unique_groupby_keys = unique_groupby_keys.sample(frac=1, random_state=seed)
    # get number of keys
    num_keys = len(unique_groupby_keys)
    # get number of keys for each split
    if sum(ratios.values()) != 1:
        raise ValueError(f"Ratios do not sum to 1: {ratios}")
    num_train = int(ratios["train"] * num_keys)
    num_val = int(ratios["val"] * num_keys)
    # split keys
    train_keys = unique_groupby_keys.iloc[:num_train]
    train_keys["split"] = "train"
    if "test" in ratios and ratios["test"] > 0:
        val_keys = unique_groupby_keys.iloc[num_train : num_train + num_val]
        val_keys["split"] = "val"
        test_keys = unique_groupby_keys.iloc[num_train + num_val :]
        test_keys["split"] = "test"
    else:
        val_keys = unique_groupby_keys.iloc[num_train:]
        val_keys["split"] = "val"
        test_keys = None
    # get split dataframe
    if test_keys is not None:
        splits_df = pd.concat([train_keys, val_keys, test_keys], axis=0)
    else:
        splits_df = pd.concat([train_keys, val_keys], axis=0)
    splits_df.reset_index(inplace=True, drop=True)
    return splits_df


def use_original_split(
    annontation_df: pd.DataFrame,
    groupby_key: list[str] | str,
    row_filter: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Uses the original split from the dataframe

    Args:
        annontation_df (pd.DataFrame): The dataframe to split.
        groupby_key (list[str] | str): The column(s) to use to group
            data that should be in the same split.
        row_filter (dict[str, list[str]] | None, optional): A dictionary
            of column names and values to filter the dataframe by.
            Defaults to None.

    Returns:
        A dataframe with a column 'split' indicating which split the
        data is in.
    """
    if isinstance(groupby_key, str):
        groupby_key = [groupby_key]
    if "original_split" not in annontation_df.columns:
        raise ValueError("original_split column not in dataframe")
    if "original_split" in groupby_key:
        raise ValueError("original_split column cannot be in groupby_key")
    if row_filter is None:
        row_filter = {}
    # filter rows
    for col, values in row_filter.items():
        annontation_df = annontation_df[annontation_df[col].isin(values)]
    # get unique keys
    unique_groupby_keys = annontation_df[groupby_key].drop_duplicates()
    # get original split
    original_split_df = annontation_df[groupby_key + ["original_split"]].drop_duplicates()
    if len(original_split_df) != len(unique_groupby_keys):
        raise ValueError("original_split does not match groupby_key")
    # get split dataframe
    splits_df = original_split_df.rename(columns={"original_split": "split"})
    splits_df.reset_index(inplace=True, drop=True)
    return splits_df


def use_fixed_split(
    annontation_df: pd.DataFrame,
    groupby_key: list[str] | str,
    split: str,
    row_filter: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Uses a fixed split from the dataframe

    Args:
        annontation_df (pd.DataFrame): The dataframe to split.
        groupby_key (list[str] | str): The column(s) to use to group
            data that should be in the same split.
        split (str): The split to use.
        row_filter (dict[str, list[str]] | None, optional): A dictionary
            of column names and values to filter the dataframe by.
            Defaults to None.

    Returns:
        A dataframe with a column 'split' indicating which split the
        data is in.
    """
    if isinstance(groupby_key, str):
        groupby_key = [groupby_key]
    if row_filter is None:
        row_filter = {}
    # filter rows
    for col, values in row_filter.items():
        annontation_df = annontation_df[annontation_df[col].isin(values)]
    # get unique keys
    unique_groupby_keys = annontation_df[groupby_key].drop_duplicates()
    # get split dataframe
    splits_df = unique_groupby_keys.copy()
    splits_df["split"] = split
    splits_df.reset_index(inplace=True, drop=True)
    return splits_df


def get_splits(
    df: pd.DataFrame,
    split_file_name: str,
    split_fn: Callable[[pd.DataFrame], pd.DataFrame],
    key: list[str] | str = "original_filepath",
):
    try:
        splits_df = read_split(split_file_name)
    except FileNotFoundError:
        splits_df = split_fn(df)
        write_split(split_file_name, splits_df)
    df = pd.merge(df, splits_df, on=key)
    return df
