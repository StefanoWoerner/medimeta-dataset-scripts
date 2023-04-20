"""Saves the Rotterdam EyePACS AIROGS dataset (train set) in the unified format.

INPUT DATA:
Expects train_labels.csv file and a folder named images/ (all zip subfolders merged)
as downloaded from https://zenodo.org/record/5793241
in ORIGINAL_DATA_PATH/AIROGS if zipped=False,
or all the files downloaded from https://zenodo.org/record/5793241 in ORIGINAL_DATA_PATH/AIROGS if zipped=True.

DATA MODIFICATIONS:
- The images are resized to out_img_size by 0-padding them to squares and resizing using the PIL library.
"""

import os
import pandas as pd
import yaml
from PIL import Image
from shutil import copyfile, rmtree
from zipfile import ZipFile
from .image_utils import zero_pad_to_square
from .paths import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "AIROGS"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "airogs"),
    info_path=os.path.join(INFO_PATH, "AIROGS.yaml"),
    out_img_size=(224, 224),
    zipped=True,
):
    assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."

    with open(info_path, "r") as f:
        info_dict = yaml.safe_load(f)

    images_rel_path = "images"
    root_path = in_path
    # extract subfolders
    if zipped:
        in_path = f"{out_path}_temp"
        os.makedirs(in_path)
        copyfile(os.path.join(root_path, "train_labels.csv"), os.path.join(in_path, "train_labels.csv"))
        # extract all subfolders
        for subfolder in [f for f in os.listdir(root_path) if f[-4:] == ".zip"]:
            with ZipFile(os.path.join(root_path, subfolder), "r") as zf:
                for zip_info in zf.infolist():
                    if zip_info.is_dir():
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)  # flattened structure
                    zf.extract(zip_info, os.path.join(in_path, images_rel_path))
        # change path to extracted folder
        root_path = in_path

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        images_paths = [
            os.path.join(images_rel_path, i_p)
            for i_p in os.listdir(os.path.join(root_path, images_rel_path))
            if i_p[-4:] == ".jpg"
        ]
        annotations = pd.read_csv(os.path.join(root_path, "train_labels.csv")).sort_values(by="challenge_id")
        annotations["original_filepath"] = annotations["challenge_id"].apply(
            lambda p: os.path.join(images_rel_path, p + ".jpg")
        )
        assert set(annotations["original_filepath"]) == set(images_paths)
        annotations["original_split"] = "train"
        annotations[info_dict["tasks"][0]["task_name"]] = annotations["class"].map({"RG": 1, "NRG": 0})
        # keep only needed columns
        annotations = annotations[["original_filepath", "original_split", info_dict["tasks"][0]["task_name"]]]

        def get_image_addannot_pair(df_row):
            """
            :param df_row: row of the annotations dataframe.
            :returns: image (PIL.Image), additional annotations (dict)"""
            path = df_row["original_filepath"]
            image = Image.open(os.path.join(root_path, path))
            add_annot = {"original_image_size": image.size}
            # transform image: pad to square, resize
            image = zero_pad_to_square(image)  # pad to square
            image.thumbnail(out_img_size, resample=Image.BICUBIC)  # resize
            return image, add_annot

        writer.write_from_dataframe(df=annotations, processing_func=get_image_addannot_pair)

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()
