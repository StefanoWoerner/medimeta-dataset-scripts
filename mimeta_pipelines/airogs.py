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
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .image_utils import zero_pad_to_square
from .paths import INFO_PATH, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "AIROGS.yaml"),
    batch_size=256,
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

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

    with UnifiedDatasetWriter(out_path, info_path, add_annot_cols=["referable_glaucoma", "original_size"]) as writer:
        images_paths = sorted(
            [
                os.path.join(images_rel_path, i_p)
                for i_p in os.listdir(os.path.join(root_path, images_rel_path))
                if i_p[-4:] == ".jpg"
            ]
        )
        annotations = pd.read_csv(os.path.join(root_path, "train_labels.csv")).sort_values(by="challenge_id")
        annotations["image_path"] = annotations["challenge_id"].apply(
            lambda p: os.path.join(images_rel_path, p + ".jpg")
        )
        # assert images_paths == annotations["images_paths"].values.tolist(), "Images paths do not match."
        annotations["class"] = annotations["class"].map({"RG": 1, "NRG": 0})
        path2lab = dict(zip(annotations["image_path"].values, annotations["class"].values))
        lab2fulllab = info_dict["tasks"][0]["labels"]

        def get_image_lab_addannot_triple(path: str):
            image = Image.open(os.path.join(root_path, path))
            label = path2lab[path]
            referable_glaucoma = lab2fulllab[label]
            add_annot = [referable_glaucoma, image.size]
            # transform image: pad to square, resize
            image = zero_pad_to_square(image)  # pad to square
            image.thumbnail(out_img_size, resample=Image.BICUBIC)  # resize
            return image, [label], add_annot

        for paths in tqdm(np.array_split(images_paths, len(images_paths) // batch_size), desc="Processing AIROGS"):
            with ThreadPool() as pool:
                imgs_labs_annots = pool.map(get_image_lab_addannot_triple, paths)
            writer.write(
                old_paths=paths,
                original_splits=["train"] * len(paths),
                task_labels=[img_lab_annot[1] for img_lab_annot in imgs_labs_annots],
                images=[img_lab_annot[0] for img_lab_annot in imgs_labs_annots],
                add_annots=[img_lab_annot[2] for img_lab_annot in imgs_labs_annots],
            )

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


def main():
    from config import config as cfg
    pipeline_name = "airogs"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
