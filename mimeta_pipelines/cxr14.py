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

import os
import pandas as pd
import tarfile
import yaml
from PIL import Image
from more_itertools import chunked
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree
from tqdm import tqdm
from zipfile import ZipFile
from .paths import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "CXR14.yaml"),
    batch_size=256,
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
        subfolder_zips = [os.path.join(images_path, f) for f in os.listdir(images_path) if f[-7:] == ".tar.gz"]

        def unzip(subfolder_zip):
            with tarfile.open(subfolder_zip, "r:gz") as tf:
                tf.extractall(os.path.dirname(images_path))
                os.remove(subfolder_zip)

        with ThreadPool(len(subfolder_zips)) as pool:
            results = pool.map(unzip, subfolder_zips)

    with UnifiedDatasetWriter(
        out_path,
        info_path,
        add_annot_cols=[
            "follow-up_nb",
            "patient_id",
            "patient_age",
            "patient_gender_f_m",
            "finding_labels",
            "view_position",
            "original_image_size",
            "original_pixel_spacing",
            "bounding_box",
        ],
    ) as writer:
        # relevant files
        # splits
        with open(os.path.join(root_path, "train_val_list.txt"), "r") as f:
            train_val_files = list(map(lambda p: p.strip("\n"), f.readlines()))
        with open(os.path.join(root_path, "test_list.txt"), "r") as f:
            test_files = list(map(lambda p: p.strip("\n"), f.readlines()))
        splits = pd.DataFrame(
            {
                "original_split": ["train"] * len(train_val_files) + ["test"] * len(test_files),
            },
            index=train_val_files + test_files,
        )
        # documentation
        for f in ("FAQ_CHESTXRAY.pdf", "LOG_CHESTXRAY.pdf", "README_CHESTXRAY.pdf"):
            copyfile(os.path.join(root_path, f), os.path.join(out_path, f"{f}_original"))
        # metadata
        metadata = pd.read_csv(os.path.join(root_path, "Data_Entry_2017_v2020.csv"), index_col="Image Index")
        possible_labels = list(info_dict["tasks"][0]["labels"].values())
        metadata["label"] = metadata["Finding Labels"].apply(
            lambda lab: [1 if l in lab else 0 for l in possible_labels]
        )
        metadata["original_image_size"] = (
            "(" + metadata["OriginalImage[Width"].astype(str) + "," + metadata["Height]"].astype(str) + ")"
        )
        metadata["original_pixel_spacing"] = (
            "(" + metadata["OriginalImagePixelSpacing[x"].astype(str) + "," + metadata["y]"].astype(str) + ")"
        )
        metadata.rename(
            columns={
                "Follow-up #": "follow-up_nb",
                "Patient ID": "patient_id",
                "Patient Age": "patient_age",
                "Patient Gender": "patient_gender",
                "View Position": "view_position",
                "Finding Labels": "finding_labels",
            },
            inplace=True,
        )
        gender_to_idx = {v: k for k, v in info_dict["tasks"][1]["labels"].items()}
        metadata["patient_gender_f_m"] = metadata["patient_gender"]
        metadata["patient_gender"] = metadata["patient_gender"].apply(lambda g: gender_to_idx[g])
        # bounding boxes
        bboxes = pd.read_csv(os.path.join(root_path, "BBox_List_2017.csv"), index_col="Image Index")
        bboxes["bounding_box"] = (
            "("
            + bboxes["Bbox [x"].astype(str)
            + ","
            + bboxes["y"].astype(str)
            + ","
            + bboxes["w"].astype(str)
            + ","
            + bboxes["h]"].astype(str)
            + ")"
        )
        add_annots = metadata[
            [
                "follow-up_nb",
                "patient_id",
                "patient_age",
                "patient_gender_f_m",
                "finding_labels",
                "view_position",
                "original_image_size",
                "original_pixel_spacing",
            ]
        ].join(bboxes[["bounding_box"]], how="left")

        def get_image_data(path: str):
            rgba = False
            image = Image.open(path)
            # some images are RGBA
            if image.mode == "RGBA":
                rgba = True
                image = image.convert("L")
            index = path.split(os.sep)[-1]
            annot = list(add_annots.loc[index])
            labels = [metadata.loc[index, "label"], metadata.loc[index, "patient_gender"]]
            original_split = splits.loc[index, "original_split"]
            # resize
            image.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
            return image, labels, original_split, annot, rgba

        all_paths = [os.path.join(images_path, p) for p in os.listdir(images_path) if p[-4:] == ".png"]
        batches = list(chunked(all_paths, batch_size))
        rgba_counter = 0
        for paths in tqdm(batches, desc="CRX14"):
            with ThreadPool() as pool:
                results = pool.map(get_image_data, paths)
            writer.write(
                old_paths=[os.path.relpath(p, root_path) for p in paths],
                original_splits=[res[2] for res in results],
                task_labels=[res[1] for res in results],
                images=[res[0] for res in results],
                add_annots=[res[3] for res in results],
            )
            rgba_counter += sum([res[4] for res in results])

        print(f"Found {rgba_counter} RGBA images, converted them.")

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


def main():
    from config import config as cfg
    pipeline_name = "cxr14"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
