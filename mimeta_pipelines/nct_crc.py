"""Saves the NCT-CRC dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the NCT-CRC-HE-100K.zip and CRC-VAL-HE-7K.zip files downloaded from
    https://zenodo.org/record/1214456
if zipped=False:
- the extracted NCT-CRC-HE-100K and CRC-VAL-HE-7K folders

DATA MODIFICATIONS:
- The images are opened and saved using PIL to remove erroneous tiff
  headers.
"""

import glob
import os
import re
from shutil import copytree, rmtree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from mimeta_pipelines.splits import make_random_split, use_fixed_split, get_splits
from .paths import INFO_PATH, folder_paths, setup
from .writer import UnifiedDatasetWriter


def _split_fn(x):
    return pd.concat(
        [
            make_random_split(
                x,
                groupby_key="original_filepath",
                ratios={"train": 0.85, "val": 0.15},
                row_filter={"original_split": ["train"]},
                seed=42,
            ),
            use_fixed_split(
                x, "original_filepath", split="test", row_filter={"original_split": ["val"]}
            ),
        ]
    )


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "NCT-CRC.yaml"),
    batch_size=256,
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

    splits = {
        "train": "NCT-CRC-HE-100K",
        "val": "CRC-VAL-HE-7K",
    }
    new_in_path = os.path.join(os.path.dirname(out_path), "NCT-CRC_temp")
    if zipped:
        for split, split_dir in splits.items():
            with ZipFile(os.path.join(in_path, f"{split_dir}.zip"), "r") as zf:
                zf.extractall(new_in_path)
    if not zipped:
        copytree(in_path, new_in_path)

    # somehow needed to avoid errors in multiprocessing
    for filepath in tqdm(
        list(glob.iglob(new_in_path + "/**/*.tif", recursive=True)),
        "NCT-CRC: loading-saving images",
    ):
        Image.fromarray(np.asarray(Image.open(filepath))).save(filepath)

    task = info_dict["tasks"][0]
    class_to_idx = {re.search(r"\((\w+)\)", v).group(1): k for k, v in task["labels"].items()}

    train_path = os.path.join(new_in_path, "NCT-CRC-HE-100K")
    tp, tl = folder_paths(root=train_path, dir_to_cl_idx=class_to_idx)
    val_path = os.path.join(new_in_path, "CRC-VAL-HE-7K")
    vp, vl = folder_paths(root=val_path, dir_to_cl_idx=class_to_idx)
    paths = tp + vp
    paths = [os.path.relpath(p, new_in_path) for p in paths]
    labs = tl + vl
    original_splits = ["train"] * len(tp) + ["val"] * len(vp)

    df = pd.DataFrame(
        zip(paths, labs, original_splits),
        columns=["original_filepath", task["task_name"], "original_split"],
    )

    # add splits to dataframe
    split_file_name = "crc_splits.csv"
    df = get_splits(df, split_file_name, _split_fn)

    def get_img_annotation_pair(df_row):
        img = Image.open(os.path.join(new_in_path, df_row["original_filepath"]))
        return img, {}

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        writer.write_from_dataframe(df=df, processing_func=get_img_annotation_pair)

    # def get_img(path: str):
    #     img = Image.open(path)
    #     return img
    #
    # task = info_dict["tasks"][0]
    #
    # with UnifiedDatasetWriter(out_path, info_path) as writer:
    #     for split, split_dir in splits.items():
    #         split_path = os.path.join(new_in_path, split_dir)
    #         class_to_idx = {
    #             re.search(r"\((\w+)\)", v).group(1): k for k, v in task["labels"].items()
    #         }
    #         batches = folder_paths(
    #             root=split_path, dir_to_cl_idx=class_to_idx, batch_size=batch_size
    #         )
    #         for paths, labs in tqdm(batches, desc=f"Processing NCT-CRC ({split} split)"):
    #             with ThreadPool() as pool:
    #                 imgs = pool.map(get_img, paths)
    #             writer.write_many(
    #                 old_paths=[os.path.relpath(p, new_in_path) for p in paths],
    #                 original_splits=[split] * len(paths),
    #                 task_labels=[{task["task_name"]: lab} for lab in labs],
    #                 add_annots=[{} for _ in labs],
    #                 images=imgs,
    #             )

    # remove temporary folder
    rmtree(new_in_path, ignore_errors=True)


def main():
    from config import config as cfg

    pipeline_name = "nct_crc"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
