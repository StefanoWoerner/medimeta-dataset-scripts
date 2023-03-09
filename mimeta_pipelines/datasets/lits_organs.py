"""Saves the Liver Tumor Segnmentation Benchmark (LiTS) Axial/Coronal/Sagittal Organ Slices datasets in the unified format.

INPUT DATA:
Expects the folders "annotations for lits", "LITS Challenge" and "LITS-Challenge-Test-Data"
as downloaded from the sources reported in the dataset_info/LiTS_organ_slices_*.yaml files,
at ORIGINAL_DATA_PATH/LITS.
If zipped=True, the subfolders in those 3 folders should be zipped (as downloaded).
If zipped=False, the contents of those 3 folders should be unzipped and flattened in the same folder
(i.e., "annotations for lits" contains the "segmentation-*.txt" and test-segmentation-*.txt" files,
"LITS Challenge" contains the "volume-*.nii" and "segmentation-*.nii" files from both batches,
and "LITS-Challenge-Test-Data" contains the "test-volume-*.nii" files from both batches).

DATA MODIFICATIONS:
- The images and masks are sliced from the original 3D volumes in axial, coronal and sagittal directions,
    and the center of the bounding box in the slice axis.
- The Hounsfield-Unit (HU) of the 3D images and are transformed into gray-scale with an abdominal window.
- The images and masks are cropped to a square, keeping the center of the bounding box and expanding the smaller side.
- The masks are converted to binary.
- The images and masks are resized to 224x224 (images with bicubic interpolation, masks taking the nearest value).

OUTPUT DATA:
- For each organ in each image, a separate image and mask is saved with the organ as label;
    this is done for each of the 3 directions (axial, coronal, sagittal), for a separate dataset.
"""

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
import yaml
from math import ceil
from multiprocessing.pool import ThreadPool
from PIL import Image
from shutil import rmtree, copytree, move
from tqdm import tqdm
from zipfile import ZipFile
from ..image_utils import slice_nii_image, ct_windowing, square_cut, Slice, draw_bounding_box, draw_colored_bounding_box
from ..paths import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH
from ..writer import UnifiedDatasetWriter


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "LITS"),
    out_paths=[
        os.path.join(UNIFIED_DATA_PATH, "lits_organs_axial"),
        os.path.join(UNIFIED_DATA_PATH, "lits_organs_coronal"),
        os.path.join(UNIFIED_DATA_PATH, "lits_organs_sagittal"),
    ],
    info_paths=[
        os.path.join(INFO_PATH, "LiTS_organ_slices_axial.yaml"),
        os.path.join(INFO_PATH, "LiTS_organ_slices_coronal.yaml"),
        os.path.join(INFO_PATH, "LiTS_organ_slices_sagittal.yaml"),
    ],
    batch_size=4,
    out_img_size=(224, 224),
    zipped=False,
):
    # Dataset preparation
    for out_path in out_paths:
        assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."

    annotations_folder = "annotations for lits"
    train_folder = "LITS Challenge"
    test_folder = "LITS-Challenge-Test-Data"

    root_path = in_path
    # extract folders
    if zipped:
        # extract to out_path (temporary: it is going to be in out_path/lits_organs_sagittal)
        in_path = f"{out_path}_temp"
        # annotations
        for split in ("training", "testing"):
            with ZipFile(os.path.join(root_path, annotations_folder, f"annotations_of_{split}_set.zip"), "r") as zf:
                zf.extractall(os.path.join(in_path, annotations_folder))
        # train
        for batch in ("1", "2"):
            dest = os.path.join(in_path, train_folder)
            with ZipFile(os.path.join(root_path, train_folder, f"Training_Batch{batch}.zip"), "r") as zf:
                zf.extractall(dest)
            copytree(
                os.path.join(dest, "media", "nas", "01_Datasets", "CT", "LITS", f"Training Batch {batch}"),
                dest,
                copy_function=move,
                dirs_exist_ok=True,
            )
            rmtree(os.path.join(dest, "media"))
        # test
        for folder in [f for f in os.listdir(os.path.join(root_path, test_folder)) if f[-4:] == ".zip"]:
            with ZipFile(os.path.join(root_path, test_folder, folder), "r") as zf:
                zf.extractall(in_path)  # folder in zip files has the same name as test_folder
    root_path = in_path

    # Annotations
    data = []
    # sort files by train/test, index
    extract_idx = lambda x: int(x.split("-")[-1].split(".")[0])
    key_fun = lambda x: (1000 if x.startswith("test") else 0) + extract_idx(x)

    # extract info from [test-]segmentation-*.txt files
    def extract_organs(file_path):
        lines = []
        with open(file_path) as f:
            for ln in f.readlines():
                t = ln.strip("\n").split(" ")
                # join bbox coordinates
                # TODO: check if this is correct
                if os.path.split(file_path)[-1] == "test-segmentation-59.txt":
                    bbox = ((int(t[2]), int(t[3])), (int(t[6]), int(t[7])), (int(t[4]), int(t[5])))
                else:
                    bbox = ((int(t[2]), int(t[3])), (int(t[4]), int(t[5])), (int(t[6]), int(t[7])))
                lines.append((t[0], t[1], bbox))
        return lines

    for annot_name in sorted(
        [p for p in os.listdir(os.path.join(root_path, annotations_folder)) if p[-4:] == ".txt"],
        key=key_fun,
    ):
        split = "test" if annot_name.startswith("test") else "train"
        idx = extract_idx(annot_name)
        img_path = os.path.join(
            train_folder if split == "train" else test_folder,
            f"{'test-' if split == 'test' else ''}volume-{idx}.nii",
        )
        mask_path = os.path.join(train_folder, f"segmentation-{idx}.nii") if split == "train" else None
        file_path = os.path.join(root_path, annotations_folder, annot_name)
        organs = extract_organs(file_path)
        for organ in organs:
            data.append([img_path, mask_path, split, *organ])
    data = pd.DataFrame(data=data, columns=["image_path", "mask_path", "split", "organ_name", "organ_key", "bbox"])

    # check class mapping well-defined
    assert (
        len(data[["organ_name", "organ_key"]].value_counts())
        == len(data["organ_name"].value_counts())
        == len(data["organ_key"].value_counts())
    )
    # TODO: stats of organs overlaps

    def create_unified_dataset(out_path, info_path, axis):
        rel_mask_path = "masks"
        rel_img_bb_path = "images_bbox"
        rel_img_bbs_path = "images_bboxes"

        def get_img_ann_pair(img_path: str, mask_path: str, bbox: tuple, file_idx: int):
            # image
            img, bbox_2d = slice_nii_image(os.path.join(root_path, img_path), bbox, axis)
            img = ct_windowing(img)
            orig_size = img.shape
            img_square = square_cut(img, bbox_2d)
            img_square = Image.fromarray(img_square)
            img_square = img_square.resize(out_img_size, resample=Image.Resampling.BICUBIC)
            # full slice with bounding box
            img_bb = draw_bounding_box(img, bbox_2d)
            img_bb = Image.fromarray(img_bb)
            out_img_bb_path_rel = os.path.join(rel_img_bb_path, f"{file_idx:06d}.tiff")
            img_bb.save(fp=os.path.join(out_path, out_img_bb_path_rel), compression=None, quality=100)
            # mask
            if mask_path:
                mask, _ = slice_nii_image(os.path.join(root_path, mask_path), bbox, axis)
                mask = square_cut(mask, bbox_2d)
                mask = Image.fromarray(mask)
                mask = mask.convert("1")
                mask = mask.resize(out_img_size, resample=Image.Resampling.NEAREST)
                out_mask_path_rel = os.path.join(rel_mask_path, f"{file_idx:06d}.tiff")
                mask.save(fp=os.path.join(out_path, out_mask_path_rel), compression=None, quality=100)
            else:
                out_mask_path_rel = ""
            return img_square, (orig_size, out_mask_path_rel)

        with UnifiedDatasetWriter(
            out_path,
            info_path,
            add_annot_cols=[
                "organ_name",
                "original_slice_size",
                "mask_filepath",
                "original_mask_filepath",
                "bboxes_image_filepath",
                "bbox",
            ],
        ) as writer:
            # class mapping
            with open(info_path, "r") as f:
                info_dict = yaml.safe_load(f)
            oldlab2newlab = {
                "heart": "heart",
                "lung-l": "left lung",
                "lung-r": "right lung",
                "liver": "liver",
                "spleen": "spleen",
                "pancreas": "pancreas",
                "kidney-l": "left kidney",
                "kidney-r": "right kidney",
                "bladder": "bladder",
                "femur-l": "left femoral head",
                "femur-r": "right femoral head",
            }
            newlab2idx = {v: k for k, v in info_dict["tasks"][0]["labels"].items()}
            oldlab2idx = {k: newlab2idx[v] for k, v in oldlab2newlab.items()}
            oldlab2color = {
                oldlab: np.array(matplotlib.cm.get_cmap("Set3")(i)[:3]) for i, oldlab in enumerate(oldlab2idx.keys())
            }
            os.makedirs(os.path.join(out_path, rel_mask_path), exist_ok=True)
            os.makedirs(os.path.join(out_path, rel_img_bb_path), exist_ok=True)
            os.makedirs(os.path.join(out_path, rel_img_bbs_path), exist_ok=True)

            # create base image for showing all bounding boxes
            for rel_img_path in data["image_path"].unique():
                img = nib.load(os.path.join(root_path, rel_img_path)).get_fdata()
                img = ct_windowing(img)
                if axis == Slice.AXIAL:
                    img = img[:, :, img.shape[2] // 2]
                elif axis == Slice.CORONAL:
                    img = img[:, img.shape[1] // 2, :]
                else:
                    img = img[img.shape[0] // 2, :, :]
                img_bbs_path = os.path.join(
                    out_path, rel_img_bbs_path, os.path.split(rel_img_path)[-1].replace(".nii", ".tiff")
                )
                img = np.array(Image.fromarray(img).convert("RGB"))
                for row in data[data["image_path"] == rel_img_path].itertuples():
                    # add bbox to full image
                    bbox = row.bbox
                    if axis == Slice.AXIAL:
                        bbox = (bbox[0], bbox[1])
                    elif axis == Slice.CORONAL:
                        bbox = (bbox[0], bbox[2])
                    else:
                        bbox = (bbox[1], bbox[2])
                    img = draw_colored_bounding_box(img, bbox, (oldlab2color[row.organ_name] * 255).astype(np.uint8))
                img = Image.fromarray(img, mode="RGB").save(img_bbs_path)
            data["bboxes_image_filepath"] = data["image_path"].apply(
                lambda x: os.path.join(rel_img_bbs_path, os.path.split(x)[-1].replace(".nii", ".tiff"))
            )
            fig, ax = plt.subplots(figsize=(4, 4))
            for i, (oldlab, color) in enumerate(oldlab2color.items()):
                ax.axhspan(i, i + 1, facecolor=color)
                ax.text(0.5, i + 0.5, oldlab2newlab[oldlab], ha="center", va="center", color="black")
            fig.savefig(os.path.join(out_path, rel_img_bbs_path, "legend.png"))

            current_idx = 0
            for batch in tqdm(
                np.array_split(data, ceil(len(data) / batch_size)),
                desc=f"Processing LITS-{axis.name}",
            ):
                image_paths, mask_paths, bboxes = batch[["image_path", "mask_path", "bbox"]].values.T
                with ThreadPool() as pool:
                    imgs_annots = pool.starmap(
                        get_img_ann_pair,
                        zip(image_paths, mask_paths, bboxes, list(range(current_idx, current_idx + len(batch)))),
                    )
                current_idx += len(batch)
                writer.write(
                    old_paths=list(image_paths),
                    original_splits=list(batch["split"].values),
                    task_labels=[[oldlab2idx[oldlab]] for oldlab in batch["organ_name"].values],
                    images=[img for img, _ in imgs_annots],
                    add_annots=list(
                        zip(
                            [oldlab2newlab[oldlab] for oldlab in batch["organ_name"].values],  # organ name
                            [annot[0] for _, annot in imgs_annots],  # original image size
                            [annot[1] for _, annot in imgs_annots],  # mask path
                            list(batch["mask_path"].values),  # original mask path
                            list(batch["bboxes_image_filepath"].values),  # bboxes image path
                            list(bboxes),  # bbox
                        )
                    ),
                )

    create_unified_dataset(out_paths[0], info_paths[0], axis=Slice.AXIAL)
    create_unified_dataset(out_paths[1], info_paths[1], axis=Slice.CORONAL)
    create_unified_dataset(out_paths[2], info_paths[2], axis=Slice.SAGITTAL)

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()
