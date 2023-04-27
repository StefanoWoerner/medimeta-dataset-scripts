"""Saves the Liver Tumor Segnmentation Benchmark (LiTS) Axial/Coronal/Sagittal Organ Slices
datasets in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
- a folder "annotations for lits" containing annotations_of_testing_set.zip
  and annotations_of_training_set.zip downloaded from
  https://ieee-dataport.org/documents/annotations-body-organ-localization-based-miccai-lits-dataset
- "LITS Challenge" downloaded from
  https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE?resourcekey=0-XIVV_7YUjB9TPTQ3NfM17A
- "LITS-Challenge-Test-Data" downloaded from
  https://drive.google.com/drive/folders/0B0vscETPGI1-NDZNd3puMlZiNWM?resourcekey=0-dZUUwJiQnUVYVpRQvs_2tQ

DATA MODIFICATIONS:
- The second and third axes of the image "test-volume-59.nii" are permuted,
  to bring it to the same format as the other images
  (and to add coeherence with the annotations).
- The images and masks are sliced from the original 3D volumes
  in the axial, coronal, and sagittal directions,
  taking the center of the bounding box in the slice plane.
- The Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale
  with an abdominal window with W=400, L=50.
- The images and masks are cropped to a square in the physical space,
  keeping the center of the bounding box and expanding the smaller side.
- The images and masks are resized to 224x224 using the PIL.Image.thumbnail method
  (images with BICUBIC interpolation, masks with NEAREST interpolation).

OUTPUT DATA:
- For each organ in each image, a separate image and mask is saved with the organ as label;
  this is done for each of the 3 directions (axial, coronal, sagittal), for a separate dataset.
"""

import os
import re
from math import ceil
from multiprocessing.pool import ThreadPool

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

from .image_utils import (
    slice_3d_image,
    ct_windowing,
    ratio_cut,
    AnatomicalPlane,
    draw_colored_bounding_box,
)
from .paths import INFO_PATH, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_paths=(
        os.path.join(INFO_PATH, "LiTS_organ_slices_axial.yaml"),
        os.path.join(INFO_PATH, "LiTS_organ_slices_coronal.yaml"),
        os.path.join(INFO_PATH, "LiTS_organ_slices_sagittal.yaml"),
    ),
    batch_size=2,
    out_img_size=(224, 224),
):
    out_paths = [setup(in_path, info_path)[1] for info_path in info_paths]

    # Base paths
    annots_folder = "annotations for lits"
    annots_train_folder = os.path.join(annots_folder, "annotations_of_training_set")
    annots_test_folder = os.path.join(annots_folder, "annotations_of_testing_set")
    train_folder = "LITS Challenge"
    test_folder = "LITS-Challenge-Test-Data"
    root_path = in_path
    masks_path = "masks"
    img_bboxes_path = "images_bboxes"

    # Dataframe with processing information
    df = _get_volumes_dataframe(root_path, train_folder, test_folder, annots_train_folder, annots_test_folder)
    df = _get_organs_dataframe(df, root_path)

    # Additional annotation columns
    add_annot_cols = [
        "organ_name",
        "original_image_size",
        "original_voxel_dims",
        "mask_filepath",
        "original_mask_filepath",
        "bboxes_image_filepath",
        "original_bbox",
    ]

    # To be called once for each plane
    def _get_unified_data(out_path, info_path, plane):
        with UnifiedDatasetWriter(out_path, info_path, add_annot_cols) as writer:
            # Get labeling transformations and legend for colored bounding boxes
            plane_df, bboxes_label_fig = _transform_labels(df.copy(), info_path)
            # Create output folders
            os.makedirs(os.path.join(out_path, masks_path), exist_ok=True)
            os.makedirs(os.path.join(out_path, img_bboxes_path), exist_ok=True)
            bboxes_label_fig.savefig(os.path.join(out_path, img_bboxes_path, "legend.png"))

            # Processing function
            def get_volume_writer_input(volume_path):
                # used for volume-related information (same for every organ)
                row = plane_df[plane_df.volume_path == volume_path].iloc[0]
                img = _load_nii_image(os.path.join(root_path, volume_path)).get_fdata()
                mask = _load_nii_image(os.path.join(root_path, row.mask_path)).get_fdata() if row.mask_path else None

                # prepare image that will show all bounding boxes
                if plane == AnatomicalPlane.AXIAL:
                    bboxes_img = img[:, :, img.shape[2] // 2]
                elif plane == AnatomicalPlane.CORONAL:
                    bboxes_img = img[:, img.shape[1] // 2, :]
                elif plane == AnatomicalPlane.SAGITTAL:
                    bboxes_img = img[img.shape[0] // 2, :, :]
                bboxes_img = zoom(ct_windowing(bboxes_img), (getattr(row, f"ratio_{plane.name.lower()}"), 1), order=3)
                bboxes_img = (bboxes_img * 255.0).astype(np.uint8)  # PIL only supports uint8 for RGB images
                bboxes_img = np.array(Image.fromarray(bboxes_img).convert("RGB"))
                bboxes_img_path = os.path.join(
                    img_bboxes_path, os.path.split(volume_path)[1].replace("volume", "bboxes").replace("nii", "tiff")
                )

                images = []
                task_labels = []
                add_annots = []

                # process organs
                for row in plane_df[plane_df["volume_path"] == volume_path].itertuples():
                    organ_img, organ_mask_img, bboxes_img = _get_organ_img_mask(
                        img, mask, bboxes_img, row, plane, out_img_size
                    )
                    if mask is not None:
                        organ_mask_img_path = os.path.join(masks_path, f"{(row.Index):06d}.tiff")
                        organ_mask_img.save(os.path.join(out_path, organ_mask_img_path))
                    else:
                        organ_mask_img_path = None
                    images.append(organ_img)
                    task_labels.append([row.new_idx])
                    add_annots.append(
                        [
                            row.new_label,  # organ_name
                            img.shape,  # original_image_size
                            row.voxel_dims,  # original_voxel_dims
                            organ_mask_img_path,  # mask_filepath
                            row.mask_path,  # original_mask_filepath
                            bboxes_img_path,  # bboxes_image_filepath
                            row.bbox,  # original_bbox
                        ]
                    )

                bboxes_img = Image.fromarray(bboxes_img, mode="RGB")
                bboxes_img.save(os.path.join(out_path, bboxes_img_path))

                old_paths = [volume_path] * len(images)
                original_splits = [row.split] * len(images)
                return old_paths, original_splits, task_labels, images, add_annots

            # Batch processing of images
            all_volume_paths = plane_df["volume_path"].unique()
            batches = np.array_split(all_volume_paths, ceil(len(all_volume_paths) / batch_size))
            for volume_paths in tqdm(batches, desc=f"Processing LITS dataset, {plane.name.lower()} plane"):
                with ThreadPool(batch_size) as pool:
                    writer_inputs = pool.map(get_volume_writer_input, volume_paths)
                # join all sublists into one list
                writer_input = [
                    [el for input in writer_inputs for el in input[i]] for i in range(len(writer_inputs[0]))
                ]
                writer.write(*writer_input)

    _get_unified_data(out_paths[0], info_paths[0], AnatomicalPlane.AXIAL)
    _get_unified_data(out_paths[1], info_paths[1], AnatomicalPlane.CORONAL)
    _get_unified_data(out_paths[2], info_paths[2], AnatomicalPlane.SAGITTAL)


def _load_nii_image(abs_path):
    # test-volume-59 has swapped axes...correct
    if os.path.split(abs_path)[1] == "test-volume-59.nii":
        nii_volume = nib.load(abs_path).as_reoriented(np.array([[0, 1], [2, 1], [1, 1]]))
    else:
        nii_volume = nib.load(abs_path)
    return nii_volume


def _get_voxeldims_ratios(volume_path):
    voxel_dims = _load_nii_image(volume_path).header.get_zooms()
    ratios = (
        voxel_dims[0] / voxel_dims[1],  # axial
        voxel_dims[0] / voxel_dims[2],  # coronal
        voxel_dims[1] / voxel_dims[2],  # sagittal
    )
    return voxel_dims, ratios


def _get_volumes_dataframe(root_path, train_folder, test_folder, annots_train_folder, annots_test_folder):
    volume_paths = []
    organs_paths = []
    mask_paths = []
    splits = []
    idxs = []
    voxel_dims = []
    ratios = [[], [], []]

    def _process_file(organs_path, split):
        if organs_path.endswith(".txt"):
            idx = int(re.findall(r"\d+", organs_path)[-1])
            if split == "train":
                batch_nb = 1 if idx <= 27 else 2
                mask_path = os.path.join(train_folder, f"Training Batch {batch_nb}", organs_path.replace("txt", "nii"))
                volume_path = mask_path.replace("segmentation", "volume")
                organs_rel_path = os.path.join(annots_train_folder, organs_path)
            else:
                volume_path = os.path.join(
                    test_folder, organs_path.replace("txt", "nii").replace("segmentation", "volume")
                )
                mask_path = None
                organs_rel_path = os.path.join(annots_test_folder, organs_path)
            volume_paths.append(volume_path)
            organs_paths.append(organs_rel_path)
            mask_paths.append(mask_path)
            splits.append(split)
            idxs.append(idx)
            _voxel_dims, _ratios = _get_voxeldims_ratios(os.path.join(root_path, volume_path))
            voxel_dims.append(_voxel_dims)
            for i in range(3):
                ratios[i].append(_ratios[i])

    for annot in os.listdir(os.path.join(root_path, annots_train_folder)):
        _process_file(annot, "train")
    for annot in os.listdir(os.path.join(root_path, annots_test_folder)):
        _process_file(annot, "test")
    df = pd.DataFrame(
        {
            "volume_path": volume_paths,
            "mask_path": mask_paths,
            "split": splits,
            "volume_idx": idxs,
            "voxel_dims": voxel_dims,
            "ratio_axial": ratios[0],
            "ratio_coronal": ratios[1],
            "ratio_sagittal": ratios[2],
        },
        index=organs_paths,
    )
    return df


def _get_organs_dataframe(volumes_df, root_path):
    data = []
    for organs_path in volumes_df.index:
        with open(os.path.join(root_path, organs_path)) as f:
            for ln in f.readlines():
                t = ln.strip("\n").split(" ")
                # join bbox coordinates
                bbox = ((int(t[2]), int(t[3])), (int(t[4]), int(t[5])), (int(t[6]), int(t[7])))
                data.append((organs_path, t[0], t[1], bbox))
    df = pd.DataFrame(data, columns=["organs_path", "old_label", "old_idx", "bbox"])
    organs_df = df.join(volumes_df, on="organs_path", how="left")
    organs_df["split_idx"] = organs_df["split"].map({"train": 0, "test": 1})
    organs_df = organs_df.sort_values(by=["split_idx", "volume_idx"]).reset_index(drop=True).drop(columns=["split_idx"])
    return organs_df


def _transform_labels(df, info_path):
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
        oldlab: (np.array(matplotlib.cm.get_cmap("Set3")(i)[:3]) * 255).astype("uint8")
        for i, oldlab in enumerate(oldlab2idx.keys())
    }
    fig, ax = plt.subplots(figsize=(4, 4))
    for i, (oldlab, color) in enumerate(oldlab2color.items()):
        ax.axhspan(i, i + 1, facecolor=color / 255)
        ax.text(0.5, i + 0.5, oldlab2newlab[oldlab], ha="center", va="center", color="black")
    df["new_label"] = df["old_label"].map(oldlab2newlab)
    df["new_idx"] = df["old_label"].map(oldlab2idx)
    df["organ_color"] = df["old_label"].map(oldlab2color)
    return df, fig


def _get_organ_img_mask(img, mask, bboxes_img, row, plane, out_img_size):
    ratio = getattr(row, f"ratio_{plane.name.lower()}")
    bbox = row.bbox
    # Organ image
    organ_img, bbox_2d = slice_3d_image(img, bbox, plane)
    organ_img = ct_windowing(organ_img)
    organ_img = (organ_img * 255).astype(np.uint8)
    organ_img = ratio_cut(organ_img, bbox_2d, ratio)
    organ_img = Image.fromarray(organ_img)
    organ_img = organ_img.resize(out_img_size, resample=Image.Resampling.BICUBIC)
    # Organ mask
    if mask is not None:
        mask_img, _ = slice_3d_image(mask, bbox, plane)
        mask_img = mask_img.astype(bool)
        mask_img = Image.fromarray(mask_img)
        mask_img = mask_img.resize(out_img_size, resample=Image.Resampling.NEAREST)
    else:
        mask_img = None
    # Bounding boxes image: draw organ bounding box
    bbox_bboxes_img = ([round(v * ratio) for v in bbox_2d[0]], bbox_2d[1])
    organ_color = row.organ_color
    bboxes_img = draw_colored_bounding_box(bboxes_img, bbox_bboxes_img, organ_color)
    return organ_img, mask_img, bboxes_img


def main():
    from config import config as cfg

    pipeline_name = "lits_organs"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
