"""Saves the Seven-Point Checklist Dermatology dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the release_v0.zip file downloaded from
  https://derm.cs.sfu.ca/Download.html
if zipped=False:
- the extracted release_v0 folder

DATA MODIFICATIONS:
- The images are zero-padded on the smallest dimension to obtain a
  square image.
- The images are resized to 224x224 using the PIL.Image.thumbnail method
  with BICUBIC interpolation.
"""

import numpy as np
import os
import pandas as pd
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from zipfile import ZipFile

from PIL import Image
from tqdm import tqdm

from .image_utils import zero_pad_to_square
from .paths import INFO_PATH, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_paths=(
        os.path.join(INFO_PATH, "derm7pt_clinic.yaml"),
        os.path.join(INFO_PATH, "derm7pt_derm.yaml"),
    ),
    batch_size=64,
    out_img_size=(224, 224),
    zipped=True,
):
    info_dicts, out_paths = zip(*[setup(in_path, info_path) for info_path in info_paths])
    root_path = in_path
    # extract folder
    if zipped:
        # extract to out_path (temporary)
        temp_path = f"{out_paths[0]}_temp"
        in_path = temp_path
        with ZipFile(os.path.join(root_path, "release_v0.zip"), "r") as zf:
            zf.extractall(in_path)
        root_path = in_path
    in_path = os.path.join(in_path, "release_v0")

    def create_unified_dataset(out_path: str, info_path: str, info_dict: dict, path_col: str):
        # Splits data
        train_split_df = pd.read_csv(os.path.join(in_path, "meta", "train_indexes.csv"), index_col="indexes")
        train_split_df["split"] = "train"
        val_split_df = pd.read_csv(os.path.join(in_path, "meta", "valid_indexes.csv"), index_col="indexes")
        val_split_df["split"] = "val"
        test_split_df = pd.read_csv(os.path.join(in_path, "meta", "test_indexes.csv"), index_col="indexes")
        test_split_df["split"] = "test"
        split_df = pd.concat([train_split_df, val_split_df, test_split_df], axis=0)
        split_df.index = split_df.index + 1  # meta.csv is 1-indexed

        # Meta data
        meta_df = pd.read_csv(os.path.join(in_path, "meta", "meta.csv"), index_col="case_num")
        meta_df = meta_df.join(split_df, how="inner")

        # Grouped labels
        meta_df["diagnosis_simplified"] = meta_df["diagnosis"].apply(lambda d: "melanoma" if "melanoma" in d else d)
        diagnosis_groups = {
            "basal cell carcinoma": "basal cell carcinoma",
            "blue nevus": "nevus",
            "clark nevus": "nevus",
            "combined nevus": "nevus",
            "congenital nevus": "nevus",
            "dermal nevus": "nevus",
            "dermatofibroma": "miscellaneous",
            "lentigo": "miscellaneous",
            "melanoma": "melanoma",
            "melanosis": "miscellaneous",
            "miscellaneous": "miscellaneous",
            "recurrent nevus": "nevus",
            "reed or spitz nevus": "nevus",
            "seborrheic keratosis": "seborrheic keratosis",
            "vascular lesion": "miscellaneous",
        }
        meta_df["diagnosis_grouped"] = meta_df["diagnosis_simplified"].map(diagnosis_groups)
        vascular_structures_groups = {
            "absent": "absent",
            "arborizing": "regular",
            "comma": "regular",
            "hairpin": "regular",
            "within regression": "regular",
            "wreath": "regular",
            "dotted": "irregular",
            "linear irregular": "irregular",
        }
        meta_df["vascular_structures_grouped"] = meta_df["vascular_structures"].map(vascular_structures_groups)
        pigmentation_groups = {
            "absent": "absent",
            "diffuse regular": "regular",
            "localized regular": "regular",
            "diffuse irregular": "irregular",
            "localized irregular": "irregular",
        }
        meta_df["pigmentation_grouped"] = meta_df["pigmentation"].map(pigmentation_groups)
        regression_structures_groups = {
            "absent": "absent",
            "blue areas": "present",
            "white areas": "present",
            "combinations": "present",
        }
        meta_df["regression_structures_grouped"] = meta_df["regression_structures"].map(regression_structures_groups)

        # Labels
        taskname2col = {
            "Diagnosis": "diagnosis_simplified",
            "Diagnosis grouped": "diagnosis_grouped",
            "Pigment Network": "pigment_network",
            "Blue Whitish Veil": "blue_whitish_veil",
            "Vascular Structures": "vascular_structures",
            "Vascular Structures grouped": "vascular_structures_grouped",
            "Pigmentation": "pigmentation",
            "Pigmentation grouped": "pigmentation_grouped",
            "Streaks": "streaks",
            "Dots and Globules": "dots_and_globules",
            "Regression Structures": "regression_structures",
            "Regression Structures grouped": "regression_structures_grouped",
        }
        tasks_mapper = {
            task["task_name"]: {lab: idx for idx, lab in task["labels"].items()} for task in info_dict["tasks"]
        }
        for task_name, col in taskname2col.items():
            assert meta_df[col].isin(tasks_mapper[task_name].keys()).all()
            meta_df[task_name] = meta_df[col].map(tasks_mapper[task_name])

        # Paths
        meta_df["original_filepath"] = meta_df[path_col].apply(lambda p: os.path.join(in_path, "images", p))
        meta_df.drop(columns=["clinic", "derm"], inplace=True)
        meta_df.set_index("original_filepath", inplace=True)

        # Add annotations
        add_annot_cols = set(meta_df.columns) - set(["split", *[task["task_name"] for task in info_dict["tasks"]]])
        add_annot_cols = list(add_annot_cols)  # to keep order

        def get_writer_input(path: str):
            df_row = meta_df.loc[path]
            # original filepath
            original_filepath = os.path.relpath(path, root_path)
            # split
            split = df_row["split"]
            # image
            img = Image.open(path)
            img_size = img.size
            # zero pad
            img = zero_pad_to_square(img)
            # resize
            img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
            # labels
            lab = {task["task_name"]: df_row[task["task_name"]] for task in info_dict["tasks"]}
            # add annotation
            add_annot = {"original_image_size": img_size, **{col: df_row[col] for col in add_annot_cols}}
            return original_filepath, split, lab, img, add_annot

        with UnifiedDatasetWriter(out_path, info_path) as writer:
            all_paths = meta_df.index
            for paths in tqdm(
                np.array_split(all_paths, len(all_paths) // batch_size), desc=f"Processing {info_dict['name']}"
            ):
                with ThreadPool() as pool:
                    writer_inputs = pool.map(get_writer_input, paths)
                writer.write_many(*(zip(*writer_inputs)))

    # create both datasets
    create_unified_dataset(out_path=out_paths[0], info_path=info_paths[0], info_dict=info_dicts[0], path_col="clinic")
    create_unified_dataset(out_path=out_paths[1], info_path=info_paths[1], info_dict=info_dicts[1], path_col="derm")

    # delete temporary folder
    if zipped:
        rmtree(temp_path)


def main():
    from config import config as cfg

    pipeline_name = "derm7pt"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
