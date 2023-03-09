"""Saves the ChestXRay14 dataset in the unified format.

INPUT DATA:
Expects zip file as downloaded from https://nihcc.app.box.com/v/ChestXray-NIHCC
at ORIGINAL_DATA_PATH/CXR14/CXR8.zip if zipped=True,
or extracted folder with the compressed subfolders extracted in place
in ORIGINAL_DATA_PATH/CXR14 if zipped=False.

DATA MODIFICATIONS:
- The images are resized to 224x224 using the PIL.Image.thumbnail method with BICUBIC interpolation.
- The 519 images in RGBA format are converted to grayscale using the PIL.Image.convert method.
"""
import glob
import os

import numpy as np
import pandas as pd
import tarfile

import pydicom
import yaml
from PIL import Image
from more_itertools import chunked
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree, copytree
from tqdm import tqdm
from zipfile import ZipFile
from .utils import (
    INFO_PATH,
    ORIGINAL_DATA_PATH,
    UNIFIED_DATA_PATH,
    UnifiedDatasetWriter,
)


ORIGINAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "original_data")
UNIFIED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "unified_data")


def sort_cbis(root_path):
    # All uncropped
    all_paths = glob.glob(os.path.join(root_path, "CBIS-DDSM", "*CC", "*", "*"))
    all_paths += glob.glob(os.path.join(root_path, "CBIS-DDSM", "*MLO", "*", "*"))
    for d in all_paths:
        fs = glob.glob(os.path.join(d, "*"))
        if len(fs) == 0:
            print(f"error on {d}: no files")
        elif len(fs) > 1:
            print(f"error on {d}: more than one file")
        else:
            # print(os.path.join(*os.path.split(fs[0])[:-1], '000000.dcm'))
            os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], "000000.dcm"))

    # All cropped and masked
    all_paths = glob.glob(os.path.join(root_path, "CBIS-DDSM", "*CC_*", "*", "*"))
    all_paths += glob.glob(os.path.join(root_path, "CBIS-DDSM", "*MLO_*", "*", "*"))
    for d in all_paths:
        fs = glob.glob(os.path.join(d, "*"))
        if len(fs) == 0:
            print(f"error on {d}: no files")
        elif len(fs) == 1:
            os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], "000000.dcm"))
        elif len(fs) > 2:
            print(f"error on {d}: more than two files")
        else:
            if os.path.getsize(fs[0]) < os.path.getsize(fs[1]):
                os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], "000000.dcm"))
                os.rename(fs[1], os.path.join(*os.path.split(fs[1])[:-1], "000001.dcm"))
            else:
                os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], "000001.dcm"))
                os.rename(fs[1], os.path.join(*os.path.split(fs[1])[:-1], "000000.dcm"))


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "CBIS-DDSM"),
    out_paths=[
        os.path.join(UNIFIED_DATA_PATH, "cbis_mass_crop"),
        os.path.join(UNIFIED_DATA_PATH, "cbis_calc_crop"),
    ],
    info_paths=[
        os.path.join(INFO_PATH, "CBIS-DDSM_mass_cropped.yaml"),
        os.path.join(INFO_PATH, "CBIS-DDSM_calc_cropped.yaml"),
    ],
    batch_size=256,
    out_img_size=(224, 224),
    is_sorted=False,
):
    new_in_path = os.path.join(UNIFIED_DATA_PATH, "cbis_temp")
    if not is_sorted:
        copytree(in_path, new_in_path)
        root_path = new_in_path
        sort_cbis(root_path)
    else:
        root_path = in_path

    _get_unified_data(
        root_path,
        out_paths[0],
        info_paths[0],
        batch_size,
        out_img_size,
        label_file_train=os.path.join(root_path, "mass_case_description_train_set.csv"),
        label_file_test=os.path.join(root_path, "mass_case_description_test_set.csv"),
    )
    _get_unified_data(
        root_path,
        out_paths[1],
        info_paths[1],
        batch_size,
        out_img_size,
        label_file_train=os.path.join(root_path, "calc_case_description_train_set.csv"),
        label_file_test=os.path.join(root_path, "calc_case_description_test_set.csv"),
    )


def _get_unified_data(
    root_path,
    out_path,
    info_path,
    batch_size,
    out_img_size,
    label_file_train,
    label_file_test,
):
    with open(info_path, "r") as f:
        info_dict = yaml.safe_load(f)

    df_train = pd.read_csv(label_file_train)
    df_test = pd.read_csv(label_file_test)
    df_train.insert(0, "original_split", "train")
    df_test.insert(0, "original_split", "test")
    df = pd.concat([df_train, df_test], axis=0)

    task_names = [task["task_name"] for task in info_dict["tasks"] if task["task_name"] != "pathology"]
    task_labels = [df["pathology"].apply(lambda x: 1 if x == "MALIGNANT" else 0).tolist()]
    for i, task_name in enumerate(task_names):
        concepts = info_dict["tasks"][i+1]["labels"].values()
        task_labels.append(df[task_name].apply(lambda x: [int(0 if pd.isna(x) else c in x) for c in concepts]).tolist())


    cropped_img_paths = df["cropped image file path"]
    cropped_img_paths = [
        os.path.join(root_path, "CBIS-DDSM", p.replace("\n", ""))
        for p in cropped_img_paths
    ]

    mask_paths = df["ROI mask file path"]
    mask_paths = [
        os.path.join(root_path, "CBIS-DDSM", p.replace("\n", "")) for p in mask_paths
    ]

    uncropped_img_paths = df["image file path"]
    uncropped_img_paths = [
        os.path.join(root_path, "CBIS-DDSM", p.replace("\n", ""))
        for p in uncropped_img_paths
    ]

    splits = df['original_split'].tolist()

    def get_image_data(i):
        s = sorted(
            (os.path.getsize(p), p)
            for p in [uncropped_img_paths[i], mask_paths[i], cropped_img_paths[i]]
        )
        dcm = pydicom.read_file(s[0][1])
        im = Image.fromarray(dcm.pixel_array.astype(np.float32) / 255)
        im = im.convert('L')
        im = im.resize(out_img_size, resample=Image.Resampling.BICUBIC)
        labels = [tl[i] for tl in task_labels]
        return s[0][1], splits[i], im, labels

    with UnifiedDatasetWriter(out_path, info_path, add_annot_cols=[]) as writer:
        for i in tqdm(range(len(df))):
            p, s, im, l = get_image_data(i)
            writer.write(
                old_paths=[p],
                original_splits=[s],
                task_labels=[l],
                images=[im],
            )

if __name__ == "__main__":
    get_unified_data()
