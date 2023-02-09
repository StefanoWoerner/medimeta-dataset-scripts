"""Saves the ChestXRay14 dataset in the unified format.

Expects zip file as downloaded from https://nihcc.app.box.com/v/ChestXray-NIHCC (if zipped=True),
or extracted folder with the compressed subfolders extracted in /images (if zipped=False),
in ORIGINAL_DATA_PATH/CXR14 named CXR14[.zip].
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
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "CXR14"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "CXR14"),
    info_path=os.path.join(INFO_PATH, "CXR14.yaml"),
    batch_size=256,
    out_img_size=(224, 224),
    zipped=False,
):
    with open(info_path, 'r') as f:
        info_dict = yaml.safe_load(f)

    with UnifiedDatasetWriter(
        out_path, info_path, add_annot_cols=[
            "follow-up_nb", "patient_id", "patient_age", "patient_gender", "view_position",
            "original_image_size", "original_pixel_spacing", "bounding_box",
        ]
    ) as writer:
        root_path = os.path.join(in_path, "CXR8")
        images_path = os.path.join(root_path, "images")
        # extract folder
        if zipped:
            with ZipFile(f"{root_path}.zip", 'r') as zf:
                zf.extractall(in_path)
            # extract subfolders
            subfolder_zips = [os.path.join(images_path, f) for f in os.listdir(images_path) if f[-7:] == ".tar.gz"]

            def unzip(subfolder_zip):
                with tarfile.open(subfolder_zip, "r:gz") as tf:
                    tf.extractall(os.path.dirname(images_path))
                    os.remove(subfolder_zip)

            with ThreadPool(len(subfolder_zips)) as pool:
                results = pool.map(unzip, subfolder_zips)

        # relevant files
        # splits
        with open(os.path.join(root_path, "train_val_list.txt"), "r") as f:
            train_val_files = list(map(lambda p: p.strip("\n"), f.readlines()))
        with open(os.path.join(root_path, "test_list.txt"), "r") as f:
            test_files = list(map(lambda p: p.strip("\n"), f.readlines()))
        splits = pd.DataFrame({
            "original_split": ["train"] * len(train_val_files) + ["test"] * len(test_files),
        }, index=train_val_files + test_files)
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
        metadata.rename(columns={
            "Follow-up #": "follow-up_nb", "Patient ID": "patient_id", "Patient Age": "patient_age",
            "Patient Gender": "patient_gender", "View Position": "view_position",
        }, inplace=True)
        # bounding boxes
        bboxes = pd.read_csv(os.path.join(root_path, "BBox_List_2017.csv"), index_col="Image Index")
        bboxes["bounding_box"] = (
            "(" + bboxes["Bbox [x"].astype(str) + "," + bboxes["y"].astype(str) + "," +
            bboxes["w"].astype(str) + "," + bboxes["h]"].astype(str) + ")"
        )
        add_annots = metadata[[
            "follow-up_nb", "patient_id", "patient_age", "patient_gender", "view_position",
            "original_image_size", "original_pixel_spacing"
        ]].join(bboxes[["bounding_box"]], how="left")

        def dataset_loader(path: str):
            rgba = False
            image = Image.open(path)
            # some images are RGBA
            if image.mode == "RGBA":
                rgba = True
                image = image.convert("L")
            index = path.split(os.sep)[-1]
            annot = list(add_annots.loc[index])
            label = metadata.loc[index, "label"]
            original_split = splits.loc[index, "original_split"]
            # resize
            image.thumbnail(out_img_size, Image.ANTIALIAS)
            return image, label, original_split, annot, rgba

        all_paths = [os.path.join(images_path, p) for p in os.listdir(images_path) if p[-4:] == ".png"]
        batches = list(chunked(all_paths, batch_size))
        n_threads = 16
        rgba_counter = 0
        for paths in tqdm(batches, desc="CRX14"):
            with ThreadPool(n_threads) as pool:
                results = pool.map(dataset_loader, paths)
            writer.write(
                old_paths=list(paths),
                original_splits=[res[2] for res in results],
                task_labels=[[res[1]] for res in results],
                images=[res[0] for res in results],
                add_annots=[res[3] for res in results],
            )
            rgba_counter += sum([res[4] for res in results])

        print("Found {} RGBA images, converted them.".format(rgba_counter))

        # remove extracted folder to free up space
        if zipped:
            rmtree(root_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()
