"""Saves the Retinal Fundus Multi-disease Image Dataset (RFMiD) dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the 'A. RFMiD_All_Classes_Dataset.zip' file downloaded from
  https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid
if zipped=False:
- the extracted 'A. RFMiD_All_Classes_Dataset' folder

DATA MODIFICATIONS:
- The images are center-cropped with the smallest dimension to obtain a
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

from .image_utils import center_crop
from .paths import INFO_PATH, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "RFMiD.yaml"),
    batch_size=128,
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

    root_path = in_path
    # extract folder
    if zipped:
        # extract to out_path (temporary)
        temp_path = f"{out_path}_temp"
        in_path = temp_path
        with ZipFile(os.path.join(root_path, "A. RFMiD_All_Classes_Dataset.zip"), "r") as zf:
            zf.extractall(in_path)
        root_path = in_path
    in_path = os.path.join(in_path, "A. RFMiD_All_Classes_Dataset")

    # Information dataframe
    csv_files_path = os.path.join(in_path, "2. Groundtruths")
    train_df = pd.read_csv(os.path.join(csv_files_path, "a. RFMiD_Training_Labels.csv"))
    train_df["split"] = "train"
    val_df = pd.read_csv(os.path.join(csv_files_path, "b. RFMiD_Validation_Labels.csv"))
    val_df["split"] = "val"
    test_df = pd.read_csv(os.path.join(csv_files_path, "c. RFMiD_Testing_Labels.csv"))
    test_df["split"] = "test"
    info_df = pd.concat([train_df, val_df, test_df])
    # Original paths
    split2splitfolder = {"train": "a. Training Set", "val": "b. Validation Set", "test": "c. Testing Set"}
    split2folder = {
        split: os.path.join(in_path, "1. Original Images", split2splitfolder[split])
        for split in ("train", "val", "test")
    }
    info_df["original_filepath"] = info_df["split"].map(split2folder) + os.path.sep + info_df["ID"].astype(str) + ".png"
    # Tasks
    risk_task, disease_task = info_dict["tasks"]
    info_df.rename(columns={"Disease_Risk": "risk"}, inplace=True)
    info_df["risk_label"] = info_df["risk"].map(risk_task["labels"])

    def label_func(row):
        labels = [None] * len(disease_task["labels"])
        for idx, disease in disease_task["labels"].items():
            labels[idx] = row[disease]
        return labels

    def diseases_func(row):
        diseases = []
        for disease in disease_task["labels"].values():
            if row[disease] == 1:
                diseases.append(disease)
        return " - ".join(diseases)

    info_df["disease"] = info_df.apply(label_func, axis=1)
    info_df["disease_labels"] = info_df.apply(diseases_func, axis=1)

    info_df.set_index("original_filepath", inplace=True, drop=True)

    def get_writer_input(path: str):
        df_row = info_df.loc[path]
        # original filepath
        original_filepath = os.path.relpath(path, root_path)
        # split
        split = df_row["split"]
        # image
        img = Image.open(path)
        img_size = img.size
        # center-crop
        img = center_crop(img)
        # resize
        img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
        # labels
        lab = [df_row["risk"], df_row["disease"]]
        # add annotation
        add_annot = [img_size, df_row["risk_label"], df_row["disease_labels"]]
        return original_filepath, split, lab, img, add_annot

    with UnifiedDatasetWriter(
        out_path, info_path, add_annot_cols=["original_size", "disease_presence", "disease_labels"]
    ) as writer:
        all_paths = info_df.index
        for paths in tqdm(np.array_split(all_paths, len(all_paths) // batch_size), desc="Processing RFMiD"):
            with ThreadPool() as pool:
                writer_inputs = pool.map(get_writer_input, paths)
            writer.write(*(zip(*writer_inputs)))

    # delete temporary folder
    if zipped:
        rmtree(temp_path)


def main():
    from config import config as cfg

    pipeline_name = "rfmid"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
