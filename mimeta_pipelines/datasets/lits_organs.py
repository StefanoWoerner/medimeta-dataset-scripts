"""Saves the Liver Tumor Segnmentation Benchmark (LiTS) Axial/Coronal/Sagittal Organ Slices datasets in the unified format.

INPUT DATA:
Expects the folders "annotations for lits", "LITS Challenge" and "LITS-Challenge-Test-Data"
as downloaded from the sources reported in the dataset_info/LiTS_organ_slices_*.yaml files,
at ORIGINAL_DATA_PATH/LITS.

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
from scipy.ndimage import zoom
from tqdm import tqdm
from ..image_utils import (
    slice_3d_image,
    ct_windowing,
    square_cut,
    Slice,
    draw_colored_bounding_box,
)
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
    batch_size=8,
    out_img_size=(224, 224),
):
    for out_path in out_paths:
        assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."

    # Base paths
    annotations_folder = "annotations for lits"
    train_folder = "LITS Challenge"
    test_folder = "LITS-Challenge-Test-Data"
    root_path = in_path
    masks_path = "masks"
    img_bboxes_path = "images_bboxes"

    # Get image paths
    test_volume_paths = []
    train_volume_paths = []
    for dir_, _, files in os.walk(root_path):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, root_path)
            rel_file = os.path.join(rel_dir, file_name)
            if "volume" in file_name and file_name.endswith(".nii"):
                if rel_file.startswith(train_folder):
                    train_volume_paths.append(rel_file)
                elif rel_file.startswith(test_folder):
                    test_volume_paths.append(rel_file)
    train_volume_paths = sorted(train_volume_paths)
    test_volume_paths = sorted(test_volume_paths)

    # Extract info from [test-]segmentation-*.txt files
    def extract_organs(file_path):
        lines = []
        with open(file_path) as f:
            for ln in f.readlines():
                t = ln.strip("\n").split(" ")
                # join bbox coordinates
                bbox = ((int(t[2]), int(t[3])), (int(t[4]), int(t[5])), (int(t[6]), int(t[7])))
                lines.append((t[0], t[1], bbox))
        return lines

    # Unstretch images and masks
    def unstretch_imgs_bboxes(volume_path):
        # Image
        nib_img = nib.load(volume_path)
        voxel_dims = nib_img.header.get_zooms()
        img = nib_img.get_fdata()
        orig_shape = img.shape
        # wrong in data...
        if "test-volume-59" in volume_path:
            img = np.swapaxes(img, 1, 2)
            voxel_dims = (voxel_dims[0], voxel_dims[2], voxel_dims[1])
        zoom_factor = np.array(voxel_dims) / voxel_dims[0]
        img = zoom(img, zoom_factor, order=3)
        # Mask
        mask_path = volume_path.replace("volume", "segmentation")
        mask_path = mask_path if os.path.exists(mask_path) else None
        if mask_path is not None:
            nib_mask = nib.load(mask_path)
            mask = np.around(zoom(nib_mask.get_fdata(), zoom_factor, order=1)).astype(bool)
        else:
            mask = None
        # Organs
        annot_path = os.path.join(
            root_path,
            annotations_folder,
            os.path.split(volume_path)[1].replace("volume", "segmentation").replace(".nii", ".txt"),
        )
        organs_out = []
        organs = extract_organs(annot_path)
        for organ in organs:
            bbox = organ[2]
            unstretched_bbox = tuple(tuple(round(zoom_factor[i] * bbox[i][j]) for j in range(2)) for i in range(3))
            organs_out.append([organ[0], organ[1], bbox, unstretched_bbox])
        return img, orig_shape, voxel_dims, mask, organs_out

    # Cut out organs
    def get_organs_masks_bboxes_imgs(img, organs, axis, oldlab2color, mask=None):
        # Base bounding boxes image
        bboxes_img = img.copy()
        if axis == Slice.AXIAL:
            bboxes_img = bboxes_img[:, :, bboxes_img.shape[2] // 2]
        elif axis == Slice.CORONAL:
            bboxes_img = bboxes_img[:, bboxes_img.shape[1] // 2, :]
        else:
            bboxes_img = bboxes_img[bboxes_img.shape[0] // 2, :, :]
        bboxes_img = ct_windowing(bboxes_img)
        bboxes_img = np.array(Image.fromarray(bboxes_img).convert("RGB"))
        organ_imgs = []
        masks = []
        for organ in organs:
            # Organ image
            bbox = organ[3]
            organ_img, bbox_2d = slice_3d_image(img, bbox, axis)
            organ_img = ct_windowing(organ_img)
            organ_img = square_cut(organ_img, bbox_2d)
            organ_img = Image.fromarray(organ_img)
            organ_img = organ_img.resize(out_img_size, resample=Image.Resampling.BICUBIC)
            organ_imgs.append(organ_img)
            # Organ mask
            if mask is not None:
                mask_sub, _ = slice_3d_image(mask, bbox, axis)
                mask_sub = square_cut(mask_sub, bbox_2d)
                mask_sub = Image.fromarray(mask_sub)
                mask_sub = mask_sub.convert("1")
                mask_sub = mask_sub.resize(out_img_size, resample=Image.Resampling.NEAREST)
            masks.append(mask_sub)
            # Bounding boxes image
            bboxes_img = draw_colored_bounding_box(bboxes_img, bbox_2d, (oldlab2color[organ[0]] * 255).astype(np.uint8))
        bboxes_img = Image.fromarray(bboxes_img, mode="RGB")
        return organ_imgs, masks, bboxes_img

    def create_unified_dataset(out_path, info_path, axis):
        with UnifiedDatasetWriter(
            out_path,
            info_path,
            add_annot_cols=[
                "organ_name",
                "original_image_size",
                "original_voxel_dims",
                "unstretched_image_size",
                "mask_filepath",
                "original_mask_filepath",
                "bboxes_image_filepath",
                "bbox",
                "bbox_unstretched",
            ],
        ) as writer:
            # Class mappings
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
            fig, ax = plt.subplots(figsize=(4, 4))
            for i, (oldlab, color) in enumerate(oldlab2color.items()):
                ax.axhspan(i, i + 1, facecolor=color)
                ax.text(0.5, i + 0.5, oldlab2newlab[oldlab], ha="center", va="center", color="black")
            os.makedirs(os.path.join(out_path, masks_path), exist_ok=True)
            os.makedirs(os.path.join(out_path, img_bboxes_path), exist_ok=True)
            fig.savefig(os.path.join(out_path, img_bboxes_path, "legend.png"))

            def process_image(volume_path, split):
                abs_volume_path = os.path.join(root_path, volume_path)
                img, orig_shape, voxel_dims, mask, organs = unstretch_imgs_bboxes(abs_volume_path)
                organ_imgs, masks, bboxes_img = get_organs_masks_bboxes_imgs(img, organs, axis, oldlab2color, mask)
                mask_paths = []
                for idx, mask in enumerate(masks):
                    mask_path = os.path.join(masks_path, os.path.split(volume_path)[-1].replace(".nii", f"_{idx}.tiff"))
                    mask_paths.append(mask_path)
                    mask.save(os.path.join(out_path, mask_path), compression=None, quality=100)
                bboxes_path = os.path.join(img_bboxes_path, os.path.split(volume_path)[-1].replace(".nii", ".tiff"))
                bboxes_img.save(os.path.join(out_path, bboxes_path), compression=None, quality=100)
                writer.write(
                    old_paths=[volume_path] * len(organs),
                    original_splits=[split] * len(organs),
                    task_labels=[[oldlab2idx[organ[0]]] for organ in organs],
                    images=organ_imgs,
                    add_annots=[
                        [
                            oldlab2newlab[organ[0]],
                            orig_shape,
                            voxel_dims,
                            img.shape,
                            mask_path,
                            volume_path.replace("volume", "segmentation") if mask_path else None,
                            bboxes_path,
                            organ[2],
                            organ[3],
                        ]
                        for organ, mask_path in zip(organs, mask_paths)
                    ],
                )
                return mask_paths

            start_idx = 0
            inputs = list(
                zip(
                    train_volume_paths + test_volume_paths,
                    ["train"] * len(train_volume_paths) + ["test"] * len(test_volume_paths),
                )
            )
            batches = np.array_split(inputs, ceil(len(inputs) / batch_size))
            for batch in tqdm(batches, desc="Processing LITS training dataset"):
                with ThreadPool(batch_size) as pool:
                    mask_paths = pool.starmap(process_image, batch)
                    # flatten
                    mask_paths = [mask_path for l in mask_paths for mask_path in l]
                    for idx, mask_path in enumerate(mask_paths):
                        os.rename(
                            os.path.join(out_path, mask_path), os.path.join(out_path, f"{(start_idx+idx):06d}.tiff")
                        )
                        start_idx += 1

    create_unified_dataset(out_paths[0], info_paths[0], axis=Slice.AXIAL)
    create_unified_dataset(out_paths[1], info_paths[1], axis=Slice.CORONAL)
    create_unified_dataset(out_paths[2], info_paths[2], axis=Slice.SAGITTAL)


if __name__ == "__main__":
    get_unified_data()
