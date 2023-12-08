"""Saves the ISIC 2018 challenge dataset (HAM10000 + validation and test
data) in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the following files as downloaded from
  https://challenge.isic-archive.com/data/
    - ISIC2018_Task3_Test_GroundTruth.zip
    - ISIC2018_Task3_Test_Input.zip
    - ISIC2018_Task3_Training_GroundTruth.zip
    - ISIC2018_Task3_Training_Input.zip
    - ISIC2018_Task3_Training_LesionGroupings.csv
    - ISIC2018_Task3_Validation_GroundTruth.zip
    - ISIC2018_Task3_Validation_Input.zip
if zipped=False:
- the ISIC2018_Task3_Test_LesionGroupings.csv file as downloaded from
    https://challenge.isic-archive.com/data/
- unzipped folders with the contents of the above zip files

DATA MODIFICATIONS:
- The images are resized to 224x224 using the PIL.Image.thumbnail method
  with BICUBIC interpolation.
"""

import os
from shutil import rmtree, copyfile
from zipfile import ZipFile

import pandas as pd
from PIL import Image
from tqdm import tqdm

from mimeta_pipelines.image_utils import center_crop
from mimeta_pipelines.paths import INFO_PATH, setup
from mimeta_pipelines.writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "HAM10000.yaml"),
    batch_size=512,
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

    root_path = in_path
    # extract folder
    if zipped:
        in_path = f"{out_path}_temp"
        os.makedirs(in_path)
        copyfile(
            os.path.join(root_path, "ISIC2018_Task3_Training_LesionGroupings.csv"),
            os.path.join(in_path, "ISIC2018_Task3_Training_LesionGroupings.csv"),
        )
        # extract all subfolders
        for zipped in [f for f in os.listdir(root_path) if f[-4:] == ".zip"]:
            with ZipFile(os.path.join(root_path, zipped), "r") as zf:
                zf.extractall(in_path)
        # change path to extracted folder
        root_path = in_path

    train_gt_path = os.path.join(
        root_path,
        "ISIC2018_Task3_Training_GroundTruth",
        "ISIC2018_Task3_Training_GroundTruth.csv",
    )
    val_gt_path = os.path.join(
        root_path,
        "ISIC2018_Task3_Validation_GroundTruth",
        "ISIC2018_Task3_Validation_GroundTruth.csv",
    )
    test_gt_path = os.path.join(
        root_path,
        "ISIC2018_Task3_Test_GroundTruth",
        "ISIC2018_Task3_Test_GroundTruth.csv",
    )

    # read in the ground truth files
    train_gt = pd.read_csv(train_gt_path, index_col=0)
    val_gt = pd.read_csv(val_gt_path, index_col=0)
    test_gt = pd.read_csv(test_gt_path, index_col=0)

    # convert the ground truth from one-hot to class indices
    train_gt["disease category"] = train_gt.idxmax(axis=1)
    val_gt["disease category"] = val_gt.idxmax(axis=1)
    test_gt["disease category"] = test_gt.idxmax(axis=1)
    train_gt["class_idx"] = train_gt["disease category"].map(train_gt.columns.get_loc)
    val_gt["class_idx"] = val_gt["disease category"].map(val_gt.columns.get_loc)
    test_gt["class_idx"] = test_gt["disease category"].map(test_gt.columns.get_loc)

    # add original split column and combine the ground truth files
    train_gt["split"] = "train"
    val_gt["split"] = "val"
    test_gt["split"] = "test"
    all_gt = pd.concat([train_gt, val_gt, test_gt])

    split_to_img_subdir = {
        "train": "ISIC2018_Task3_Training_Input",
        "val": "ISIC2018_Task3_Validation_Input",
        "test": "ISIC2018_Task3_Test_Input",
    }

    with UnifiedDatasetWriter(out_path, info_path) as writer:

        def get_from_df(i):
            row = all_gt.iloc[i]
            orig_split = row.split
            img_path = os.path.join(root_path, split_to_img_subdir[orig_split], f"{row.name}.jpg")
            img = Image.open(img_path)
            orig_size = img.size
            img = center_crop(img)
            img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
            annot = {"original_image_size": orig_size}
            lab = {info_dict["tasks"][0]["task_name"]: row["class_idx"]}
            return img_path, orig_split, img, annot, lab

        for j in tqdm(range(0, len(all_gt))):
            original_path, original_split, image, annotations, label = get_from_df(j)

            writer.write_many(
                old_paths=[os.path.relpath(original_path, root_path)],
                splits=[original_split],
                original_splits=[original_split],
                task_labels=[label],
                images=[image],
                add_annots=[annotations],
            )

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


def main():
    from config import config as cfg

    pipeline_name = "ham10000"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
