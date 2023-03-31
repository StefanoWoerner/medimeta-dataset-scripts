"""Saves the peripheral blood cells dataset in the unified format.

INPUT DATA:
Expects zip file as downloaded from
https://data.mendeley.com/datasets/snkd93bnjr/1
at ORIGINAL_DATA_PATH/peripheral_blood_cells/PBC_dataset_normal_DIB.zip if zipped=True,
or extracted folder in ORIGINAL_DATA_PATH/peripheral_blood_cells/PBC_dataset_normal_DIB if zipped=False.

DATA MODIFICATIONS:
- The images are center-cropped with the smallest dimension to obtain a square image
    (only some are slightly non-square).
- The images are resized to 224x224 using the PIL.Image.thumbnail method with BICUBIC interpolation.
"""

import os
import yaml
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from tqdm import tqdm
from zipfile import ZipFile
from .image_utils import center_crop
from .paths import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, folder_paths
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "peripheral_blood_cells"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "pbc"),
    info_path=os.path.join(INFO_PATH, "peripheral_blood_cells.yaml"),
    batch_size=512,
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
        with ZipFile(os.path.join(root_path, "PBC_dataset_normal_DIB.zip"), "r") as zf:
            zf.extractall(in_path)
    # data path
    root_path = os.path.join(in_path, "PBC_dataset_normal_DIB")

    def get_img_annotation_pair(path: str):
        img = Image.open(path)
        # center-crop
        img, w, h = center_crop(img)
        # resize
        img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
        # add annotation
        add_annot = [(w, h)]
        return img, add_annot

    # rename ig -> immature granulocyte
    os.rename(os.path.join(root_path, "ig"), os.path.join(root_path, "immature granulocyte"))

    with UnifiedDatasetWriter(out_path, info_path, add_annot_cols=["original_size", "cell_class_label"]) as writer:
        class_to_idx = {v: k for k, v in info_dict["tasks"][0]["labels"].items()}
        batches = folder_paths(
            root=root_path, batch_size=batch_size, dir_to_cl_idx=class_to_idx, check_alphabetical=False
        )
        for paths, labs in tqdm(batches, desc="Processing peripheral_blood_cells dataset"):
            with ThreadPool() as pool:
                imgs_annots = pool.map(get_img_annotation_pair, paths)
            writer.write(
                old_paths=[os.path.relpath(p, root_path) for p in paths],
                original_splits=["train"] * len(paths),
                task_labels=[[lab] for lab in labs],
                images=[img_annot[0] for img_annot in imgs_annots],
                add_annots=[
                    img_annot[1] + [info_dict["tasks"][0]["labels"][lab]] for img_annot, lab in zip(imgs_annots, labs)
                ],
            )

    # delete temporary folder
    if zipped:
        rmtree(in_path)
    # leave directory structure as found in original data
    else:
        os.rename(os.path.join(root_path, "immature granulocyte"), os.path.join(root_path, "ig"))


if __name__ == "__main__":
    get_unified_data()
