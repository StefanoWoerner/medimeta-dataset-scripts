"""Saves the peripheral blood cells dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the PBC_dataset_normal_DIB.zip file downloaded from
    https://data.mendeley.com/datasets/snkd93bnjr/1
if zipped=False:
- the extracted PBC_dataset_normal_DIB folder

DATA MODIFICATIONS:
- The images are center-cropped with the smallest dimension to obtain a
  square image (only some are slightly non-square).
- The images are resized to 224x224 using the PIL.Image.thumbnail method
  with BICUBIC interpolation.
"""

import os
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from zipfile import ZipFile

from PIL import Image
from tqdm import tqdm

from .image_utils import center_crop
from .paths import INFO_PATH, folder_paths, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "peripheral_blood_cells.yaml"),
    batch_size=512,
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

    root_path = in_path
    # extract folder
    if zipped:
        # extract to out_path (temporary)
        in_path = f"{out_path}_temp"
        with ZipFile(os.path.join(root_path, "PBC_dataset_normal_DIB.zip"), "r") as zf:
            zf.extractall(in_path)
    # data path
    root_path = os.path.join(in_path, "PBC_dataset_normal_DIB")

    def get_img_annotation_pair(path: str):
        img = Image.open(path)
        # center-crop
        w, h = img.size
        img = center_crop(img)
        # resize
        img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
        # add annotation
        add_annot = {"original_image_size": (w, h)}
        return img, add_annot

    # rename ig -> immature granulocyte
    os.rename(os.path.join(root_path, "ig"), os.path.join(root_path, "immature granulocyte"))

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        task = info_dict["tasks"][0]
        class_to_idx = {v: k for k, v in task["labels"].items()}
        batches = folder_paths(
            root=root_path, dir_to_cl_idx=class_to_idx, batch_size=batch_size, check_alphabetical=False
        )
        for paths, labs in tqdm(batches, desc="Processing peripheral_blood_cells dataset"):
            with ThreadPool() as pool:
                imgs_annots = pool.map(get_img_annotation_pair, paths)
            writer.write_many(
                old_paths=[os.path.relpath(p, root_path) for p in paths],
                original_splits=["train"] * len(paths),
                task_labels=[{task["task_name"]: lab} for lab in labs],
                images=[img_annot[0] for img_annot in imgs_annots],
                add_annots=[img_annot[1] for img_annot in imgs_annots],
            )

    # delete temporary folder
    if zipped:
        rmtree(in_path)
    # leave directory structure as found in original data
    else:
        os.rename(os.path.join(root_path, "immature granulocyte"), os.path.join(root_path, "ig"))


def main():
    from config import config as cfg

    pipeline_name = "pbc"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
