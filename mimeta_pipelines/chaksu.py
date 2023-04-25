"""Saves the Chákṣu dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the Readme_Chaksu IMAGE Database.pdf file downloaded from
  https://figshare.com/articles/dataset/Ch_k_u_A_glaucoma_specific_fundus_image_database/20123135
- the Train.zip file (same source)
- the Test.zip file (same source)
if zipped=False:
- the Readme_Chaksu IMAGE Database.pdf file downloaded from
  (https://figshare.com/articles/dataset/Ch_k_u_A_glaucoma_specific_fundus_image_database/20123135)
- the Train folder (Train.zip extracted, same source)
- the Test folder (Test.zip extracted, same source)

DATA MODIFICATIONS:
- The images are zero-padded to square shape and resized, using the
  PIL.Image.thumbnail method with BICUBIC interpolation.
"""

import os
import re
from functools import partial
from shutil import copyfile, rmtree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from .image_utils import zero_pad_to_square
from .paths import INFO_PATH, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "Chaksu.yaml"),
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

    # Dataset specific namings
    readme_name = "Readme_Chaksu IMAGE Database.pdf"
    split_folders = ["Train", "Test"]
    devices = ["Forus", "Remidio", "Bosch"]
    expert_idxs = list(range(1, 6))
    merge_algos = ("Majority", "Mean", "Median", "STAPLE")
    stats = ("Majority", "Mean", "Median")

    root_path = in_path
    # Extract subfolders
    if zipped:
        in_path = f"{out_path}_temp"
        os.makedirs(in_path)
        copyfile(os.path.join(root_path, readme_name), os.path.join(in_path, readme_name))
        # extract all subfolders
        for split_folder in split_folders:
            with ZipFile(os.path.join(root_path, split_folder + ".zip"), "r") as zf:
                zf.extractall(in_path)
        # change path to extracted folder
        root_path = in_path

    # Paths in output data
    cup_masks_path = os.path.join("masks", "cup")
    disc_masks_path = os.path.join("masks", "disc")

    # Create dataframe with all info
    images_df = _get_images_df(root_path, split_folders, devices)
    masks_df = _get_masks_df(root_path, images_df["original_filepath"], expert_idxs, merge_algos)
    labels_df = _get_labels_df(root_path, split_folders, devices, expert_idxs, stats)
    info_df = images_df.join(masks_df, on="original_filepath", how="left")
    info_df = info_df.join(labels_df, on="image_name", how="left")
    info_df.drop_duplicates(inplace=True)
    info_df.reset_index(drop=True, inplace=True)

    # Remove not needed columns
    rem_cols = [f"mask_expert_{exp_idx}_path" for exp_idx in expert_idxs]
    rem_cols += [f"mask_algo_{merge_algo}_path" for merge_algo in set(merge_algos) - set(["STAPLE"])]
    info_df.drop(columns=rem_cols, inplace=True)

    # Map annotations to labels
    task = info_dict["tasks"][0]
    annot2lab = {"NORMAL": "Normal", "GLAUCOMA SUSPECT": "Suspect", "GLAUCOMA  SUSUPECT": "Suspect"}  # typo in data
    lab2idx = {v: k for k, v in task["labels"].items()}
    annot2idx = {k: lab2idx[v] for k, v in annot2lab.items()}
    info_df[task["task_name"]] = info_df["annot_majority"].map(annot2idx)

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        os.makedirs(os.path.join(out_path, cup_masks_path))
        os.makedirs(os.path.join(out_path, disc_masks_path))
        copyfile(os.path.join(root_path, readme_name), os.path.join(out_path, readme_name))

        def get_image_addannot_pair(df_row):
            # get info from dataframe
            image_path = df_row["original_filepath"]
            mask_path = df_row["mask_algo_STAPLE_path"]
            # transform image
            img = Image.open(os.path.join(root_path, image_path))
            orig_size = img.size
            img = zero_pad_to_square(img)
            img.thumbnail(out_img_size, resample=Image.BICUBIC)
            # transform masks
            mask = Image.open(os.path.join(root_path, mask_path))
            mask = zero_pad_to_square(mask)
            cup_mask = Image.fromarray((np.array(mask) > 255 // 3).astype(bool))
            cup_mask.thumbnail(out_img_size, resample=Image.NEAREST)
            cup_mask_path = os.path.join(cup_masks_path, writer.image_name_from_index(df_row["index"]))
            cup_mask.save(os.path.join(out_path, cup_mask_path))
            disc_mask = Image.fromarray((np.array(mask) > (255 * 2) // 3).astype(bool))
            disc_mask.thumbnail(out_img_size, resample=Image.NEAREST)
            disc_mask_path = os.path.join(disc_masks_path, writer.image_name_from_index(df_row["index"]))
            disc_mask.save(os.path.join(out_path, disc_mask_path))
            add_annot = {"original_image_size": orig_size}
            return img, add_annot

        writer.write_from_dataframe(df=info_df, processing_func=get_image_addannot_pair)

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


def _get_image_name(path: str) -> str:
    """Name of the file without path to parent directory and extension.
    Needed because between image path, mask path, and annotation image name,
    the image file extensions in the original data are often different.
    """
    return os.path.splitext(os.path.basename(path).lower())[0]


def _get_image_path(root_path, dir_path, image_name):
    """Reverse mapping from _get_image_name; reconstructs the path (without extension)
    using root_path and dir_path, and looks for a file with that name in the obtained directory.
    Fails the assert if there are more than one file (or none) with that name.
    """
    # special case where segmentation is named differently
    if "4.0_OD_CO" in dir_path and "Expert 4" in dir_path and image_name in ("22", "82", "83"):
        image_name = image_name + "glsusp"
    image_paths = [
        path for path in os.listdir(os.path.join(root_path, dir_path)) if _get_image_name(path) == image_name
    ]
    assert (
        len(image_paths) == 1
    ), f"Image path not (uniquely) identifiable from image name {image_name} (in {os.path.join(root_path, dir_path)}) -> {image_paths}"
    return os.path.join(dir_path, image_paths[0])


def _get_images_df(root_path, split_folders, devices):
    """Builds a dataframe with the image paths of the original images in the original dataset."""
    splits_ = []
    devices_ = []
    image_paths = []
    image_names = []
    for split in split_folders:
        for device in devices:
            dir_path = os.path.join(split, "1.0_Original_Fundus_Images", device)
            paths = [
                os.path.join(dir_path, path)
                for path in os.listdir(os.path.join(root_path, dir_path))
                if path[-4:] in (".JPG", ".jpg", ".png")
            ]
            image_names += [_get_image_name(path) for path in paths]
            image_paths += paths
            splits_ += [split.lower()] * len(paths)
            devices_ += [device] * len(paths)
    images_df = pd.DataFrame(
        {"original_split": splits_, "device": devices_, "original_filepath": image_paths, "image_name": image_names}
    )
    # patient id implicit in the naming of some images
    images_df["patient_id"] = images_df["original_filepath"].apply(
        lambda p: getattr(re.search(r"P(\d+)", p), "group", lambda: None)()
    )
    return images_df


def _get_masks_df(root_path, image_paths, expert_idxs, merge_algos):
    """Builds a dataframe with the masks paths corresponding to the passed image_paths.
    It looks for the masks for each expert annotator and algorithm merging annotations from all the expert annotators.
    """
    masks_df = pd.DataFrame({"original_filepath": image_paths})

    def _extract_mask_path(image_path: str, expert_idx=None, merge_algo=None):
        assert (expert_idx is None) != (merge_algo is None)  # exactly one specified
        assert (merge_algo is None) or (merge_algo in merge_algos)  # allowed value
        assert (expert_idx is None) or (expert_idx in expert_idxs)  # allowed value
        # extract info from image path
        split, _, device, _ = image_path.split(os.path.sep)
        image_name = _get_image_name(image_path)
        base_path = os.path.join(split, "4.0_OD_CO_Fusion_Images")
        if expert_idx:
            mask_dir_path = os.path.join(base_path, f"Expert {expert_idx}", device)
        else:
            # adapt to typo in the original data
            if device == "Remidio" and split == "Test":
                device = "Remedio"
            mask_dir_path = os.path.join(base_path, device, merge_algo)
        mask_path = _get_image_path(root_path, mask_dir_path, image_name)
        return mask_path

    # masks drawn by experts
    for expert_idx in expert_idxs:
        masks_df[f"mask_expert_{expert_idx}_path"] = masks_df["original_filepath"].apply(
            partial(_extract_mask_path, expert_idx=expert_idx)
        )
    # merging of masks drawn by experts
    for merge_algo in merge_algos:
        masks_df[f"mask_algo_{merge_algo}_path"] = masks_df["original_filepath"].apply(
            partial(_extract_mask_path, merge_algo=merge_algo)
        )
    masks_df.set_index("original_filepath", drop=True, inplace=True)

    return masks_df


def _get_labels_df(root_path, split_folders, devices, expert_idxs, stats):
    """Builds a dataframe containing the annotations (suspect glaucoma/normal) from all experts,
    as well as merged annotations (majority/...).
    """
    labels_dfs = []

    def reindex_df(df):
        # weird naming in original data
        df["image_name"] = df["Images"].apply(lambda n: os.path.splitext(n.lower().split("-")[0])[0])
        df.drop(columns=["Images"], inplace=True)
        df.set_index("image_name", drop=True, inplace=True)
        return df

    for split in split_folders:
        for device in devices:
            dev_lab_df = pd.read_csv(
                os.path.join(
                    root_path, split, "6.0_Glaucoma_Decision", f"Glaucoma_Decision_Comparison_{device}_majority.csv"
                )
            )
            dev_lab_df.rename(
                columns={
                    "Glaucoma Decision": "annot_majority",
                    "Majority Decision": "annot_majority",  # inconsistent naming in source data
                    **{f"Expert.{idx}": f"annot_expert_{idx}" for idx in expert_idxs},
                },
                inplace=True,
            )
            reindex_df(dev_lab_df)
            dev_info_dfs = []
            for stat in stats:
                stat_df = pd.read_csv(
                    os.path.join(root_path, split, "6.0_Glaucoma_Decision", stat, f"{device}.csv"), index_col="Images"
                )
                stat_df.rename(columns=lambda c: f"{c.replace(' ', '_')}_{stat}", inplace=True)
                stat_df.reset_index(inplace=True)
                reindex_df(stat_df)
                dev_info_dfs.append(stat_df)
            dev_info_df = pd.concat(dev_info_dfs, axis=1, join="inner")
            dev_lab_df = dev_lab_df.join(dev_info_df, how="inner")
            labels_dfs.append(dev_lab_df)
    labels_df = pd.concat(labels_dfs)
    return labels_df


def main():
    from config import config as cfg

    pipeline_name = "chaksu"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
