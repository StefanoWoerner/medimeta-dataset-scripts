"""Saves the Kermany OCT and Chest X-Ray datasets in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the ZhangLabData.zip file downloaded from
  https://data.mendeley.com/datasets/rscbjbr9sj/3
if zipped=False:
- the extracted CellData folder containing the subfolders OCT and
  chest_xray

DATA MODIFICATIONS for OCT:
- The images are padded to obtain a square image.
- The images are resized to 224x224 using the PIL.Image.thumbnail method
  with BICUBIC interpolation.

DATA MODIFICATIONS for Pneumonia:
- The 283 images in RGB format are converted to grayscale using the
  PIL.Image.convert method.
- The images are padded to obtain a square image.
- The images are resized to 224x224 (some upsized, since smaller than
  224x224) using the PIL.Image.resize method with BICUBIC interpolation.
"""

import glob
import os
from shutil import rmtree
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from mimeta_pipelines.splits import get_splits, make_random_split, use_fixed_split
from .image_utils import zero_pad_to_square
from .paths import INFO_PATH, setup, UNIFIED_DATA_BASE_PATH
from .writer import UnifiedDatasetWriter


_groupby_key = "patient_id"


def _split_fn(x):
    return pd.concat(
        [
            make_random_split(
                x,
                groupby_key=_groupby_key,
                ratios={"train": 0.85, "val": 0.15},
                row_filter={"original_split": ["train"]},
                seed=42,
            ),
            use_fixed_split(
                x, _groupby_key, split="test", row_filter={"original_split": ["test"]}
            ),
        ]
    )


def _get_data(in_path):
    def _get_data_from_path(path: str):
        original_filepath = os.path.relpath(path, in_path)
        original_split, class_, filename = original_filepath.split(os.path.sep)
        subclass, patient_id, _ = filename.split("-")
        return original_filepath, original_split, class_, subclass, int(patient_id)

    all_paths = glob.glob(os.path.join(in_path, "*", "*", "*.jpeg"))
    all_paths = sorted(all_paths)
    data = [_get_data_from_path(p) for p in all_paths if not os.path.basename(p).startswith(".")]
    df = pd.DataFrame(
        data, columns=["original_filepath", "original_split", "class", "subclass", "patient_id"]
    )
    return df


def get_unified_oct_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "Kermany_OCT.yaml"),
    out_img_size=(224, 224),
):
    info_dict, out_path = setup(in_path, info_path)

    df = _get_data(in_path)

    # Task mappers
    # disease class task
    class_task = info_dict["tasks"][0]
    class_to_idx = {v: k for k, v in class_task["labels"].items()}
    # urgent referral task
    referral_task = info_dict["tasks"][1]
    referral_to_idx = {v: k for k, v in referral_task["labels"].items()}
    class_to_referral = {"NORMAL": "NO", "DRUSEN": "NO", "CNV": "YES", "DME": "YES"}
    class_to_referral_idx = {c: referral_to_idx[class_to_referral[c]] for c in class_to_idx}

    df[class_task["task_name"]] = df["class"].map(class_to_idx)
    df[referral_task["task_name"]] = df["class"].map(class_to_referral_idx)

    # add splits to dataframe
    split_file_name = "OCT_splits.csv"
    df = get_splits(df, split_file_name, _split_fn, _groupby_key)

    # This counter is used to keep track of the number of RGB images
    # that are converted to grayscale. This counter is not thread-safe,
    # but it is only used for logging purposes.
    rgb_counter = 0

    def get_img_annotation_pair(df_row):
        nonlocal rgb_counter
        img = Image.open(os.path.join(in_path, df_row["original_filepath"]))
        # some images are RGB
        if img.mode == "RGB":
            rgb_counter += 1
            img = img.convert("L")
        w, h = img.size
        # pad to square
        img = zero_pad_to_square(img)
        # resize
        img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
        # add annotation
        add_annot = {"original_image_size": (w, h), "original_image_ratio": w / h}
        return img, add_annot

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        writer.write_from_dataframe(df=df, processing_func=get_img_annotation_pair)

    print(f"Converted {rgb_counter} RGB images to grayscale.")


def get_unified_pneumonia_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "Kermany_Pneumonia.yaml"),
    out_img_size=(224, 224),
):
    info_dict, out_path = setup(in_path, info_path)

    df = _get_data(in_path)

    # Task mappers
    # pneumonia presence task
    class_task = info_dict["tasks"][0]
    class_to_idx = {v: k for k, v in class_task["labels"].items()}
    # disease class task
    subclass_task = info_dict["tasks"][1]
    subclass_to_idx = {v: k for k, v in subclass_task["labels"].items()}

    df[class_task["task_name"]] = df["class"].map(class_to_idx)
    df[subclass_task["task_name"]] = df["subclass"].map(subclass_to_idx)

    # add splits to dataframe
    split_file_name = "Pneumonia_splits.csv"
    df = get_splits(df, split_file_name, _split_fn, _groupby_key)

    # This counter is used to keep track of the number of RGB images
    # that are converted to grayscale. This counter is not thread-safe,
    # but it is only used for logging purposes.
    rgb_counter = 0

    def get_img_annotation_pair(df_row):
        nonlocal rgb_counter
        img = Image.open(os.path.join(in_path, df_row["original_filepath"]))
        # some images are RGB
        if img.mode == "RGB":
            rgb_counter += 1
            img = img.convert("L")
        w, h = img.size
        # pad to square
        img = zero_pad_to_square(img)
        # resize
        img = img.resize(out_img_size, resample=Image.BICUBIC)  # resize
        # add annotation
        add_annot = {"original_image_size": (w, h), "original_image_ratio": w / h}
        return img, add_annot

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        writer.write_from_dataframe(df=df, processing_func=get_img_annotation_pair)

    print(f"Converted {rgb_counter} RGB images to grayscale.")


def get_unified_data(
    in_path,
    info_paths=(
        os.path.join(INFO_PATH, "Kermany_OCT.yaml"),
        os.path.join(INFO_PATH, "Kermany_Pneumonia.yaml"),
    ),
    out_img_size=(224, 224),
    zipped=True,
):
    root_path = in_path
    # extract folder
    if zipped:
        # extract to temporary path
        in_path = os.path.join(UNIFIED_DATA_BASE_PATH, "kermany_cell_temp")
        with ZipFile(os.path.join(root_path, "ZhangLabData.zip"), "r") as zf:
            zf.extractall(in_path)
    # data paths
    root_path = os.path.join(in_path, "CellData")
    oct_path = os.path.join(root_path, "OCT")
    pneumonia_path = os.path.join(root_path, "chest_xray")

    get_unified_oct_data(oct_path, info_paths[0], out_img_size)
    get_unified_pneumonia_data(pneumonia_path, info_paths[1], out_img_size)

    # delete temporary folder
    if zipped:
        rmtree(in_path)


def main():
    from config import config as cfg

    pipeline_name = "kermany_cell_data"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
