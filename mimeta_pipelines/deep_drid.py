"""Saves the DeepDRid regular fundus and ultra-widefield datasets in the unified format.

INPUT DATA:
Expects zip file as downloaded from https://isbi.deepdr.org/
at ORIGINAL_DATA_PATH/DeepDRiD/DeepDRiD-master.zip if zipped=True,
or extracted folder at ORIGINAL_DATA_PATH/DeepDRiD/DeepDRiD-master if zipped=False.

DATA MODIFICATIONS:
- The images are center-cropped with the smallest dimension to obtain a square image.
- The images are resized to 224x224 using the PIL.Image.thumbnail method with BICUBIC interpolation.
"""

import os
import pandas as pd
import pathlib
import yaml
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import rmtree, copyfile
from tqdm import tqdm
from zipfile import ZipFile
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter, center_crop


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "DeepDRiD"),
    out_paths=[
        os.path.join(UNIFIED_DATA_PATH, "DeepDRiD_regular-fundus"),
        os.path.join(UNIFIED_DATA_PATH, "DeepDRiD_ultra-widefield"),
    ],
    info_paths=[
        os.path.join(INFO_PATH, "DeepDRiD_regular-fundus.yaml"),
        os.path.join(INFO_PATH, "DeepDRiD_ultra-widefield.yaml"),
    ],
    batch_size=128,
    out_img_size=(224, 224),
    zipped=True,
):
    for out_path in out_paths:
        assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."

    root_path = in_path
    # extract folder
    if zipped:
        # extract to out_path (temporary: it is going to be in out_path/DeepDRid_ultra_widefield_temp)
        in_path = f"{out_path}_temp"
        with ZipFile(os.path.join(root_path, "DeepDRiD-master.zip"), "r") as zf:
            zf.extractall(in_path)
    root_path = os.path.join(in_path, "DeepDRiD-master")
    dataset_paths = [
        os.path.join(root_path, dir_name) for dir_name in ["regular_fundus_images", "ultra-widefield_images"]
    ]

    # define annotation columns and task-dataset column mappings
    reg_annot_cols = [
        "patient_id",
        "image_id",
        "left_eye_dr_level",
        "right_eye_dr_level",
        "source",
    ]
    reg_tasks_mapper = {
        orig_name: orig_name.lower().replace(" ", "_")
        for orig_name in ("DR level", "Overall quality", "Artifact", "Clarity", "Field definition")
    }
    ultra_annot_cols = ["patient_id", "image_id", "position"]
    ultra_tasks_mapper = {"DR level": "dr_level"}

    # similar work for regular fundus and ultrawidefield
    for dataset_path, out_path, info_path, annot_cols, tasks_mapper in zip(
        dataset_paths,
        out_paths,
        info_paths,
        (reg_annot_cols, ultra_annot_cols),
        (reg_tasks_mapper, ultra_tasks_mapper),
    ):
        # add one column per task with human-readable labels
        annot_cols += [task_name + "_label" for task_name in tasks_mapper.values()]

        with open(info_path, "r") as f:
            info_dict = yaml.safe_load(f)

        # image converter + annotation extractor
        def get_img_annotation_pair(path: str):
            img = Image.open(path)
            # center-crop
            img, w, h = center_crop(img)
            # resize
            img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
            # add annotation: original size and ratio
            add_annot = [(w, h), w / h]
            return img, add_annot

        # write dataset
        with UnifiedDatasetWriter(
            out_path,
            info_path,
            # sorted for consistency with below
            add_annot_cols=sorted(annot_cols) + ["original_size", "original_ratio"],
        ) as writer:
            # folder name -> split name
            split_mapper = {"training": "train", "validation": "val", "Evaluation": "test"}
            # split path -> split name
            splits = {
                os.path.join(dataset_path, path): [v for k, v in split_mapper.items() if k in path][0]
                for path in os.listdir(dataset_path)
            }
            for split_path, split in splits.items():
                # copy in readmes (sometimes .txt, sometimes .docx)
                readme_name = [f for f in os.listdir(split_path) if f.lower().startswith("readme")][0]
                copyfile(
                    os.path.join(split_path, readme_name),
                    os.path.join(out_path, f"Readme_{split}.{readme_name.split('.')[-1]}"),
                )
                # info csv; special handling for test set (less information)
                if split != "test":
                    # merge source info and other info
                    info_files = [f for f in os.listdir(split_path) if f.endswith(".csv")]
                    annots_df = (
                        pd.concat(
                            [
                                pd.read_csv(
                                    os.path.join(split_path, info_file),
                                    index_col=["patient_id", "image_id", "image_path"],
                                )
                                for info_file in info_files
                            ],
                            axis=1,
                            join="inner",
                        )
                        .reset_index(level=["patient_id", "image_id"])  # index by image_path only
                        .rename(columns=lambda n: n.lower().replace(" ", "_"))  # normalize column names
                        .rename(columns={"patient_dr_level": "dr_level"})  # coherence with ultra-widefield dataset
                    )

                    # Reindex by actual image path
                    def get_path(path):
                        path = os.path.join(
                            # assigned paths start with unexisting folder
                            split_path,
                            "Images",
                            *pathlib.Path(path.replace("\\", "/").strip("/")).parts[1:],
                        )
                        # some files wrongly named (no patient_id prefixed in name)
                        if not os.path.exists(path):
                            path_head, path_tail = os.path.split(path)
                            path_tail = path_tail.split("_")[1]
                            path = os.path.join(path_head, path_tail)
                        return path

                    annots_df.index = [get_path(path) for path in annots_df.index]

                else:  # test split
                    info_files = [f for f in os.listdir(split_path) if f.endswith(".xlsx")]
                    annots_df = (
                        pd.concat(
                            [
                                pd.read_excel(os.path.join(split_path, info_file), index_col="image_id")
                                for info_file in info_files
                            ],
                            axis=1,
                            join="inner",
                        ).rename(columns=lambda n: n.lower().replace(" ", "_"))
                        # make naming consistent with other splits
                        .rename(columns={"dr_levels": "dr_level", "uwf_dr_levels": "dr_level"})
                    )
                    # No additional columns -> fill missing
                    for col in annot_cols:
                        annots_df[col] = ""
                    annots_df["image_id"] = annots_df.index
                    annots_df["patient_id"] = annots_df.index.str.split("_").str[0]
                    # Reindex by actual image path
                    annots_df.index = [
                        os.path.join(split_path, "Images", i.split("_")[0], i + ".jpg") for i in annots_df.index
                    ]

                # mapper task column -> idx_to_label dict
                task_labels = {tasks_mapper[task["task_name"]]: task["labels"] for task in info_dict["tasks"]}
                # labeled task columns
                for col in tasks_mapper.values():
                    annots_df[col + "_label"] = annots_df[col].map(task_labels[col])

                # write (batched)
                for batch in tqdm(
                    [annots_df[pos : pos + batch_size] for pos in range(0, len(annots_df), batch_size)],
                    desc=f"Processing DeepDRiD-{os.path.split(dataset_path)[1]} ({split} split)",
                ):
                    with ThreadPool() as pool:
                        images_annots = pool.map(get_img_annotation_pair, batch.index)
                    writer.write(
                        old_paths=[os.path.relpath(p, root_path) for p in batch.index],
                        original_splits=[split] * len(batch),
                        task_labels=batch[list(tasks_mapper.values())].values.tolist(),
                        images=[img_annot[0] for img_annot in images_annots],
                        add_annots=[
                            df_annot + img_annot[1]
                            for df_annot, img_annot in zip(batch[sorted(annot_cols)].values.tolist(), images_annots)
                        ],
                    )
    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()