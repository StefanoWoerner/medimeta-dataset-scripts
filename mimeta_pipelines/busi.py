"""Saves the dataset of Breast Ultrasound Images in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the Dataset_BUSI.zip file downloaded from
  https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
if zipped=False:
- the extracted Dataset_BUSI_with_GT folder downloaded from
    https://scholar.cu.edu.eg/?q=afahmy/pages/dataset

DATA MODIFICATIONS:
- The images are converted to grayscale, the masks to binary, using the
  PIL.Image.convert method.
- The images and masks are center-cropped with the smallest dimension to
  obtain a square image
- The images are resized to 224x224 (1 upscaled) using the
  PIL.Image.thumbnail method with BICUBIC interpolation, the masks with
  NEAREST interpolation.
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
    info_path=os.path.join(INFO_PATH, "BUSI.yaml"),
    batch_size=512,
    out_img_size=(224, 224),
    zipped=False,
):
    info_dict, out_path = setup(in_path, info_path)

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
        w, h = img.size
        img = center_crop(img)
        mask = center_crop(mask)
        # resize
        if img.size[0] < out_img_size[0]:
            print("Upscaled")
        img = img.resize(out_img_size, resample=Image.Resampling.BICUBIC)
        mask = mask.resize(out_img_size, resample=Image.Resampling.NEAREST)  # binary mask (could change to max)
        # save mask
        assert len(mask.getbands()) == 1
        assert mask.mode == "1"  # binary
        out_mask_path_rel = writer.save_image_from_index(mask, file_idx, rel_masks_path)
        # add annotations
        add_annot = {
            "mask_path": out_mask_path_rel,
            "original_mask_path": os.path.relpath(mask_path, root_path),
            "original_image_size": (w, h),
        }
        return img, add_annot

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        os.makedirs(os.path.join(out_path, rel_masks_path))
        # 3-class task
        task = info_dict["tasks"][0]
        class_to_idx = {v: k for k, v in task["labels"].items()}
        # binary task
        task_bin = info_dict["tasks"][1]
        class_to_idx_bin = {v: k for k, v in task_bin["labels"].items()}
        # mapper between the two tasks
        class_to_bin = {
            "normal": "no malignant finding",
            "benign": "no malignant finding",
            "malignant": "malignant finding",
        }

        batches = folder_paths(
            root=root_path, dir_to_cl_idx=class_to_idx, batch_size=batch_size, check_alphabetical=False
        )
        current_idx = 0
        for paths, labs in tqdm(batches, desc="Processing BUSI dataset"):
            # filter out mask images
            labs = [lab for path, lab in zip(paths, labs) if "mask" not in path]
            paths = [path for path in paths if "mask" not in path]
            with ThreadPool() as pool:
                imgs_annots = pool.starmap(
                    get_img_annotation_pair, zip(paths, list(range(current_idx, current_idx + len(paths))))
                )
            current_idx += len(paths)
            # named labels
            named_labs = [task["labels"][lab] for lab in labs]
            named_labs_bin = [class_to_bin[n_lab] for n_lab in named_labs]
            # numeric label for binary task
            labs_bin = [class_to_idx_bin[n_lab_bin] for n_lab_bin in named_labs_bin]
            writer.write_many(
                old_paths=[os.path.relpath(p, root_path) for p in paths],
                original_splits=["train"] * len(paths),
                task_labels=[
                    {task["task_name"]: lab, task_bin["task_name"]: lab_bin} for lab, lab_bin in zip(labs, labs_bin)
                ],
                images=[img_annot[0] for img_annot in imgs_annots],
                add_annots=[img_annot[1] for img_annot in imgs_annots],
            )

    # delete temporary folder
    if zipped:
        rmtree(in_path)


def main():
    from config import config as cfg

    pipeline_name = "busi"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
