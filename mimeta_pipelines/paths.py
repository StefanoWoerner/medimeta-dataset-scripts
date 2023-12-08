"""Utilities for paths handling.
"""
import os

import yaml

import config

# Base paths
INFO_PATH = config.config["dataset_info_dir"]
UNIFIED_DATA_BASE_PATH = config.config["unified_data_base_path"]


def setup(in_path, info_path):
    assert os.path.exists(in_path), f"Input path {in_path} does not exist."
    assert os.path.exists(info_path), f"Info path {info_path} does not exist."
    with open(info_path, "r") as f:
        info_dict = yaml.safe_load(f)
    out_path = os.path.join(UNIFIED_DATA_BASE_PATH, info_dict["id"])

    assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."
    return info_dict, out_path


def folder_paths(
    root: str,
    dir_to_cl_idx: dict[str, int],
    batch_size: int | None = None,
    check_alphabetical: bool = True,
    check_cl_idxs_range: bool = True,
) -> tuple[list[str], list[int]] | list[tuple[list[str], list[int]]]:
    """Get batches of (paths, labels), or (paths, labels), from a folder class structure.

    Args:
        root: root folder.
        dir_to_cl_idx: dictionary mapping directories to class indices.
        batch_size: batch size if batches are to be returned, else None.
        check_alphabetical: check that the class names are in
            alphabetical order, and the indices range(len(classes)).
        check_cl_idxs_range: check that the class keys (indices) are
            equal to range(len(classes)).
    
    Returns:
        list of batches, each batch is a tuple of (paths, labels), or a
        single tuple of (paths, labels) if batch_size is None.
    """
    # alphabetical class order check
    if check_alphabetical:
        assert sorted(dir_to_cl_idx.items(), key=lambda x: x[1]) == sorted(dir_to_cl_idx.items(), key=lambda x: x[0])
    # class indices range check
    if check_cl_idxs_range:
        assert sorted(dir_to_cl_idx.values()) == list(range(len(dir_to_cl_idx)))
    # get paths and labels
    paths = []
    labels = []
    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    assert set(dirs) == set(dir_to_cl_idx.keys())  # class_dict correct and complete
    for dir_ in dirs:
        new_paths = sorted(
            [
                os.path.join(root, dir_, f)
                for f in os.listdir(os.path.join(root, dir_))
                if os.path.splitext(f)[1].lower() in (".tif", ".tiff", ".png", ".jpeg", ".jpg")
                and not f.startswith(".")
            ]
        )
        paths.extend(new_paths)
        labels.extend([dir_to_cl_idx[dir_]] * len(new_paths))
    if batch_size is None:
        return paths, labels
    else:
        batches = [(paths[i : i + batch_size], labels[i : i + batch_size]) for i in range(0, len(paths), batch_size)]
        return batches
