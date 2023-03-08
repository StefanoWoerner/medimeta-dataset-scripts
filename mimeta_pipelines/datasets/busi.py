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
import yaml
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from tqdm import tqdm
from zipfile import ZipFile
from ..image_utils import center_crop
from ..paths import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, folder_paths
from ..writer import UnifiedDatasetWriter


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "BUSI"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "busi"),
    info_path=os.path.join(INFO_PATH, "BUSI.yaml"),
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
        with ZipFile(os.path.join(root_path, "Dataset_BUSI.zip"), "r") as zf:
            zf.extractall(in_path)
    # data path
    root_path = os.path.join(in_path, "Dataset_BUSI_with_GT")

    rel_masks_path = "masks"

    def get_img_annotation_pair(path: str, file_idx: int):
        name, extension = os.path.splitext(path)
        mask_path = f"{name}_mask{extension}"
        img = Image.open(path)
        mask = Image.open(mask_path)
        # convert image to grayscale
        img = img.convert("L")
        # convert mask to binary
        mask = mask.convert("1")
        # center-crop
        img, w, h = center_crop(img)
        mask, _, _ = center_crop(mask)
        # resize
        if img.size[0] < out_img_size[0]:
            print("Upscaled")
        img = img.resize(out_img_size, resample=Image.Resampling.BICUBIC)
        mask = mask.resize(out_img_size, resample=Image.Resampling.NEAREST)  # binary mask (could change to max)
        # add annotations
        out_mask_path_rel = os.path.join(rel_masks_path, f"{file_idx:06d}.tiff")
        add_annot = [
            out_mask_path_rel,
            os.path.relpath(mask_path, root_path),
            (w, h),
        ]
        # save mask
        assert len(mask.getbands()) == 1
        assert mask.mode == "1"  # binary
        mask.save(fp=os.path.join(out_path, out_mask_path_rel), compression=None, quality=100)
        return img, add_annot

    with UnifiedDatasetWriter(
        out_path,
        info_path,
        add_annot_cols=["mask_path", "original_mask_path", "original_size", "case_label", "malignancy_label"],
    ) as writer:
        os.makedirs(os.path.join(out_path, rel_masks_path))
        # 3-class task
        class_to_idx = {v: k for k, v in info_dict["tasks"][0]["labels"].items()}
        # binary task
        class_to_idx_bin = {v: k for k, v in info_dict["tasks"][1]["labels"].items()}
        # mapper between the two tasks
        class_to_bin = {"normal": "benign", "benign": "benign", "malignant": "malignant"}

        batches = folder_paths(
            root=root_path, batch_size=batch_size, dir_to_cl_idx=class_to_idx, check_alphabetical=False
        )
        current_idx = 0
        for paths, labs in tqdm(batches, desc="Processing peripheral_blood_cells dataset"):
            # filter out mask images
            labs = [lab for path, lab in zip(paths, labs) if "mask" not in path]
            paths = [path for path in paths if "mask" not in path]
            with ThreadPool() as pool:
                imgs_annots = pool.starmap(
                    get_img_annotation_pair, zip(paths, list(range(current_idx, current_idx + len(paths))))
                )
            current_idx += len(paths)
            # named labels
            named_labs = [info_dict["tasks"][0]["labels"][lab] for lab in labs]
            named_labs_bin = [class_to_bin[n_lab] for n_lab in named_labs]
            # numeric label for binary task
            labs_bin = [class_to_idx_bin[n_lab_bin] for n_lab_bin in named_labs_bin]
            writer.write(
                old_paths=[os.path.relpath(p, root_path) for p in paths],
                original_splits=["train"] * len(paths),
                task_labels=[[lab, lab_bin] for lab, lab_bin in zip(labs, labs_bin)],
                images=[img_annot[0] for img_annot in imgs_annots],
                add_annots=[
                    img_annot[1] + [n_lab, n_lab_bin]
                    for img_annot, n_lab, n_lab_bin in zip(imgs_annots, named_labs, named_labs_bin)
                ],
            )

    # delete temporary folder
    if zipped:
        rmtree(in_path)


if __name__ == "__main__":
    get_unified_data()
