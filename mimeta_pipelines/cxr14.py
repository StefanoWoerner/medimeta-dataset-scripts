"""Saves the ChestXRay14 dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the CXR8.zip compressed folder downloaded from https://nihcc.app.box.com/v/ChestXray-NIHCC
if zipped=False:
- the images/ folder with the contents of images/images_xxx.tar.gz extracted (xxx: 001-012)
  downloaded from https://nihcc.app.box.com/v/ChestXray-NIHCC
- the Data_Entry_2017.csv file (same source)
- the FAQ_CHESTXRAY.pdf file (same source)
- the LOG_CHESTXRAY.pdf file (same source)
- the README_CHESTXRAY.pdf file (same source)
- the train_val_list.txt file (same source)
- the test_list.txt file (same source)
- the BBox_List_2017.csv file (same source)

DATA MODIFICATIONS:
- The images are resized to 224x224 using the PIL.Image.thumbnail method with BICUBIC interpolation.
- The 519 images in RGBA format are converted to grayscale using the PIL.Image.convert method.
"""

import os
import tarfile
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from mimeta_pipelines.splits import get_splits, make_random_split, use_fixed_split
from .paths import INFO_PATH, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "CXR14.yaml"),
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

    root_path = in_path
    images_path = os.path.join(root_path, "images")
    # extract folder
    if zipped:
        # extract to out_path (temporary)
        in_path = f"{out_path}_temp"
        with ZipFile(os.path.join(root_path, "CXR8.zip"), "r") as zf:
            zf.extractall(in_path)
        # change path to extracted folder
        root_path = os.path.join(in_path, "CXR8")
        images_path = os.path.join(root_path, "images")
        # extract subfolders
        subfolder_zips = [
            os.path.join(images_path, f) for f in os.listdir(images_path) if f[-7:] == ".tar.gz"
        ]

        def unzip(subfolder_zip):
            with tarfile.open(subfolder_zip, "r:gz") as tf:
                tf.extractall(os.path.dirname(images_path))
                os.remove(subfolder_zip)

        with ThreadPool(len(subfolder_zips)) as pool:
            pool.map(unzip, subfolder_zips)

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        # relevant files
        # splits
        with open(os.path.join(root_path, "train_val_list.txt"), "r") as f:
            train_val_files = list(map(lambda p: p.strip("\n"), f.readlines()))
        with open(os.path.join(root_path, "test_list.txt"), "r") as f:
            test_files = list(map(lambda p: p.strip("\n"), f.readlines()))
        path2split = {p: "train" for p in train_val_files}
        path2split.update({p: "test" for p in test_files})
        # documentation
        for f in ("FAQ_CHESTXRAY.pdf", "LOG_CHESTXRAY.pdf", "README_CHESTXRAY.pdf"):
            copyfile(os.path.join(root_path, f), os.path.join(out_path, f"{f}_original"))
        # metadata
        metadata = pd.read_csv(
            os.path.join(root_path, "Data_Entry_2017_v2020.csv"), index_col="Image Index"
        )
        metadata["original_split"] = metadata.index.map(path2split)
        task = info_dict["tasks"][0]
        possible_labels = list(task["labels"].values())
        metadata[task["task_name"]] = metadata["Finding Labels"].apply(
            lambda lab: [1 if l in lab else 0 for l in possible_labels]
        )
        metadata["original_image_size"] = (
            "("
            + metadata["OriginalImage[Width"].astype(str)
            + ","
            + metadata["Height]"].astype(str)
            + ")"
        )
        metadata["original_pixel_spacing"] = (
            "("
            + metadata["OriginalImagePixelSpacing[x"].astype(str)
            + ","
            + metadata["y]"].astype(str)
            + ")"
        )
        task_gender = info_dict["tasks"][1]
        metadata.rename(
            columns={
                "Follow-up #": "follow-up_nb",
                "Patient ID": "patient_id",
                "Patient Age": "patient_age",
                "Patient Gender": task_gender["task_name"],
                "View Position": "view_position",
                "Finding Labels": "finding_labels",
            },
            inplace=True,
        )
        gender2idx = {v: k for k, v in task_gender["labels"].items()}
        metadata[task_gender["task_name"]] = metadata[task_gender["task_name"]].map(gender2idx)
        # bounding boxes
        bboxes = pd.read_csv(os.path.join(root_path, "BBox_List_2017.csv"))
        bboxes["bounding_box"] = (
            bboxes["Finding Label"]
            + ": ("
            + bboxes["Bbox [x"].astype(str)
            + ","
            + bboxes["y"].astype(str)
            + ","
            + bboxes["w"].astype(str)
            + ","
            + bboxes["h]"].astype(str)
            + ")"
        )
        bboxes = pd.DataFrame(bboxes.groupby("Image Index")["bounding_box"].apply(list))
        bboxes.rename(columns={0: "bounding_boxes"}, inplace=True)
        info_df = metadata[
            [
                "original_split",
                task["task_name"],
                task_gender["task_name"],
                "follow-up_nb",
                "patient_id",
                "patient_age",
                "finding_labels",
                "view_position",
                "original_image_size",
                "original_pixel_spacing",
            ]
        ].join(bboxes[["bounding_box"]], how="left")
        info_df["original_filepath"] = [
            os.path.join(os.path.relpath(images_path, root_path), ind) for ind in info_df.index
        ]
        info_df.reset_index(inplace=True, drop=True)

        def split_fn(x):
            return pd.concat(
                [
                    make_random_split(
                        x,
                        groupby_key="patient_id",
                        ratios={"train": 0.85, "val": 0.15},
                        row_filter={"original_split": ["train"]},
                        seed=42,
                    ),
                    use_fixed_split(
                        x,
                        groupby_key="patient_id",
                        split="test",
                        row_filter={"original_split": ["test"]},
                    ),
                ]
            )

        info_df = get_splits(info_df, "CXR14.csv", split_fn, "patient_id")

        def get_image_addannot_pair(df_row):
            image = Image.open(os.path.join(root_path, df_row["original_filepath"]))
            # some images are RGBA
            if image.mode == "RGBA":
                image = image.convert("L")
            # resize
            image.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
            return image, dict()

        writer.write_from_dataframe(df=info_df, processing_func=get_image_addannot_pair)

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


def main():
    from config import config as cfg

    pipeline_name = "cxr14"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
