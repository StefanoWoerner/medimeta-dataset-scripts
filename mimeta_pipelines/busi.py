"""Saves the dataset of Breast Ultrasound Images in the unified format.

INPUT DATA:
Expects zip file as downloaded from https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
at ORIGINAL_DATA_PATH/BUSI/Dataset_BUSI.zip if zipped=True,
or extracted folder in ORIGINAL_DATA_PATH/BUSI/Dataset_BUSI_with_GT if zipped=False.

DATA MODIFICATIONS:
- The images are converted to grayscale, the masks to binary, using the PIL.Image.convert method.
- The images and masks are center-cropped with the smallest dimension to obtain a square image
- The images are resized to 224x224 (1 upscaled) using the PIL.Image.thumbnail method with BICUBIC interpolation,
    the masks with NEAREST interpolation.
"""

import os
import pandas as pd
import yaml
from PIL import Image
from shutil import rmtree
from zipfile import ZipFile
from .image_utils import center_crop
from .paths import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, folder_paths
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "BUSI"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "busi"),
    info_path=os.path.join(INFO_PATH, "BUSI.yaml"),
    out_img_size=(224, 224),
    zipped=True,
):
    assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."

    with open(info_path, "r") as f:
        info_dict = yaml.safe_load(f)

    root_path = in_path
    # extract folder
    if zipped:
        # extract to out_path (temporary)
        in_path = f"{out_path}_temp"
        with ZipFile(os.path.join(root_path, "Dataset_BUSI.zip"), "r") as zf:
            zf.extractall(in_path)
    # data path
    root_path = os.path.join(in_path, "Dataset_BUSI_with_GT")

    with UnifiedDatasetWriter(
        out_path,
        info_path,
        add_annot_cols=["mask_path", "original_mask_path", "original_size", "case_label", "malignancy_label"],
    ) as writer:
        rel_masks_path = "masks"
        os.makedirs(os.path.join(out_path, rel_masks_path))

        class_task = info_dict["tasks"][0]
        class2idx = {class_: idx for idx, class_ in class_task.items()}
        paths, labels = folder_paths(root=root_path, dir_to_cl_idx=class2idx)
        annot_df = pd.DataFrame(data={"original_filepath": paths, class_task["task_name"]: labels})
        # remove masks
        annot_df = annot_df[~(annot_df["original_filepath"].str.contains("mask"))]
        # 3-class task to binary task
        bin_task = info_dict["tasks"][1]
        class2bin = {"normal": "benign", "benign": "benign", "malignant": "malignant"}
        bin2binidx = {bin_: idx for idx, bin_ in bin_task["labels"].items()}
        classidx2binidx = {classidx: bin2binidx[class2bin[class_]] for classidx, class_ in class_task["labels"].items()}
        annot_df[bin_task["task_name"]] = annot_df[class_task["task_name"]].map(classidx2binidx)

        def get_img_addannot_pair(df_row):
            path = os.path.join(root_path, df_row.original_filepath)
            name, extension = os.path.splitext(path)
            mask_path = f"{name}_mask{extension}"
            img = Image.open(path)
            mask = Image.open(mask_path)
            # convert image to grayscale
            img = img.convert("L")
            # convert mask to binary
            mask = mask.convert("1")
            # center-crop
            orig_size = img.size
            img = center_crop(img)
            mask = center_crop(mask)
            # resize
            if img.size[0] < out_img_size[0]:
                print("Upscaled")
            img = img.resize(out_img_size, resample=Image.Resampling.BICUBIC)
            mask = mask.resize(out_img_size, resample=Image.Resampling.NEAREST)  # binary mask (could change to max)
            # add annotations
            out_mask_path_rel = os.path.join(rel_masks_path, writer.image_name_from_index(df_row.index))
            add_annot = {
                "mask_path": out_mask_path_rel,
                "original_mask_path": os.path.relpath(mask_path, root_path),
                "original_image_size": orig_size,
            }
            # save mask
            assert len(mask.getbands()) == 1
            assert mask.mode == "1"  # binary
            mask.save(fp=os.path.join(out_path, out_mask_path_rel), compression=None, quality=100)
            return img, add_annot

        writer.write_from_dataframe(df=annot_df, processing_func=get_img_addannot_pair)

    # delete temporary folder
    if zipped:
        rmtree(in_path)


if __name__ == "__main__":
    get_unified_data()
