"""Saves the CBIS-DDSM dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if sorted=False (default):
- the manifest-ZkhPvrLo5216730872708713142 folder downloaded from the
  TCIA using the downloader
if sorted=True:
- the properly sorted and renamed contents of the
  manifest-ZkhPvrLo5216730872708713142 folder

DATA MODIFICATIONS:
- The ROIs (bounding boxes) are extended to a minimum size of 224x224
- The bounding boxes are then squared by extending the shorter side to
  the length of the longer side
- Region crops are extracted from the full images using the extended
  bounding boxes
- The region crops are resized to 224x224 using PIL.Image.resize with
  BICUBIC interpolation.
- In contrast to the cropped images provided by CBIS-DDSM, the croppped
  images are not brightness-adjusted. The crops contain the original
  brightness levels from the full images.
"""
import glob
import os
from shutil import copytree, rmtree

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

from .splits import get_splits, make_random_split, use_fixed_split
from .image_utils import ratio_cut
from .paths import INFO_PATH, UNIFIED_DATA_BASE_PATH, setup
from .writer import UnifiedDatasetWriter


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
    in_path,
    info_paths=(
        os.path.join(INFO_PATH, "CBIS-DDSM_mass_cropped.yaml"),
        os.path.join(INFO_PATH, "CBIS-DDSM_calc_cropped.yaml"),
    ),
    out_img_size=(224, 224),
    is_sorted=False,
    remove_temp=True,
):
    new_in_path = os.path.join(UNIFIED_DATA_BASE_PATH, "cbis_temp")
    if not is_sorted:
        manifest_dirname = "manifest-ZkhPvrLo5216730872708713142"
        copytree(os.path.join(in_path, manifest_dirname), new_in_path)
        root_path = new_in_path
        sort_cbis(root_path)
    else:
        root_path = in_path

    mass_annotation_columns = {
        "patient_id": "patient_id",
        "breast_density": "breast_density",
        "left or right breast": "left_or_right_breast",
        "image view": "image_view",
        "abnormality id": "abnormality_id",
        "abnormality type": "abnormality_type",
        "assessment": "assessment",
        "subtlety": "subtlety",
    }

    calc_annotation_columns = {
        "patient_id": "patient_id",
        "breast density": "breast_density",
        "left or right breast": "left_or_right_breast",
        "image view": "image_view",
        "abnormality id": "abnormality_id",
        "abnormality type": "abnormality_type",
        "assessment": "assessment",
        "subtlety": "subtlety",
    }

    _get_unified_data(
        root_path,
        info_paths[0],
        out_img_size,
        label_file_train=os.path.join(root_path, "mass_case_description_train_set.csv"),
        label_file_test=os.path.join(root_path, "mass_case_description_test_set.csv"),
        annotation_columns=mass_annotation_columns,
    )
    _get_unified_data(
        root_path,
        info_paths[1],
        out_img_size,
        label_file_train=os.path.join(root_path, "calc_case_description_train_set.csv"),
        label_file_test=os.path.join(root_path, "calc_case_description_test_set.csv"),
        annotation_columns=calc_annotation_columns,
    )

    # delete temporary folder
    if not is_sorted and remove_temp:
        rmtree(new_in_path)


def _get_unified_data(
    root_path,
    info_path,
    out_img_size,
    label_file_train,
    label_file_test,
    annotation_columns,
):
    info_dict, out_path = setup(root_path, info_path)

    df_train = pd.read_csv(label_file_train)
    df_test = pd.read_csv(label_file_test)
    df_train.insert(0, "original_split", "train")
    df_test.insert(0, "original_split", "test")
    df = pd.concat([df_train, df_test], axis=0)

    task_names = [
        task["task_name"] for task in info_dict["tasks"] if task["task_name"] != "pathology"
    ]
    task_labels = [df["pathology"].apply(lambda x: 1 if x == "MALIGNANT" else 0).tolist()]
    for i, task_name in enumerate(task_names):
        concepts = info_dict["tasks"][i + 1]["labels"].values()
        task_labels.append(
            df[task_name]
            .apply(lambda x: [int(0 if pd.isna(x) else c in x) for c in concepts])
            .tolist()
        )

    cropped_img_paths = df["cropped image file path"]
    cropped_img_paths = [
        os.path.join(root_path, "CBIS-DDSM", p.replace("\n", "")) for p in cropped_img_paths
    ]

    mask_paths = df["ROI mask file path"]
    mask_paths = [os.path.join(root_path, "CBIS-DDSM", p.replace("\n", "")) for p in mask_paths]

    uncropped_img_paths = df["image file path"]
    uncropped_img_paths = [
        os.path.join(root_path, "CBIS-DDSM", p.replace("\n", "")) for p in uncropped_img_paths
    ]

    annotations = df[list(annotation_columns.keys())]

    split_file_name = f"{info_dict['id']}_splits.csv"

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

    df = get_splits(df, split_file_name, split_fn, "patient_id")

    splits = df["split"].tolist()
    original_splits = df["original_split"].tolist()

    def get_image_data(i):
        s = sorted(
            (os.path.getsize(p), p)
            for p in [uncropped_img_paths[i], mask_paths[i], cropped_img_paths[i]]
        )
        # find complete image, mask
        imgs = [pydicom.read_file(p[1]).pixel_array for p in s[1:]]
        if len(np.unique(imgs[0])) == 2:
            mask = imgs[0]
            im = imgs[1]
        else:
            mask = imgs[1]
            im = imgs[0]
        # bounding box from mask
        non_0_rows = np.argwhere(np.any(mask, axis=1))
        left_bb = np.min(non_0_rows)
        right_bb = np.max(non_0_rows)
        non_0_cols = np.argwhere(np.any(mask, axis=0))
        top_bb = np.min(non_0_cols)
        bottom_bb = np.max(non_0_cols)
        # extend bbox to 224x224
        size_horizontal = right_bb - left_bb
        size_vertical = bottom_bb - top_bb
        if size_horizontal < 224:
            left_bb = left_bb - (224 - size_horizontal) // 2
            right_bb = right_bb + (224 - size_horizontal) // 2
        if size_vertical < 224:
            top_bb = top_bb - (224 - size_vertical) // 2
            bottom_bb = bottom_bb + (224 - size_vertical) // 2
        # crop image
        im = ratio_cut(im, ((left_bb, right_bb), (top_bb, bottom_bb)), ratio=1.0)
        im = Image.fromarray(
            ((im.astype(np.float32) / np.iinfo(im.dtype).max) * 255).astype(np.uint8)
        )
        im = im.resize(out_img_size, resample=Image.Resampling.BICUBIC)
        labels = {
            "pathology": task_labels[0][i],
            **{tn: tl[i] for tn, tl in zip(task_names, task_labels[1:])},
        }
        original_filepath = os.path.relpath(s[0][1], root_path)
        return (
            original_filepath,
            splits[i],
            original_splits[i],
            im,
            labels,
            annotations.iloc[i].to_dict(),
        )

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        for i in tqdm(range(len(df))):
            p, s, orig_s, im, l, a = get_image_data(i)
            writer.write(
                old_path=p,
                split=s,
                original_split=orig_s,
                task_labels=l,
                add_annots=a,
                image=im,
            )


def main():
    from config import config as cfg

    pipeline_name = "cbis"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
