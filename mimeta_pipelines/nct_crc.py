"""Saves the NCT-CRC dataset in the unified format.

INPUT DATA:
Expects zip files as downloaded from https://zenodo.org/record/1214456
at ORIGINAL_DATA_PATH/NCT-CRC/NCT-CRC-HE-100K.zip and CRC-VAL-HE-7K.zip if zipped=True,
or extracted folders in ORIGINAL_DATA_PATH/NCT-CRC/NCT-CRC-HE-100K and ORIGINAL_DATA_PATH/NCT-CRC/CRC-VAL-HE-7K
if zipped=False.

DATA MODIFICATIONS:
None.
"""

import os
import re
import yaml
from shutil import rmtree
from tqdm import tqdm
from zipfile import ZipFile
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter, folder_paths


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "NCT-CRC"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "NCT-CRC"),
    info_path=os.path.join(INFO_PATH, "NCT-CRC.yaml"),
    batch_size=2048,
    zipped=True,
):
    assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."

    with open(info_path, "r") as f:
        info_dict = yaml.safe_load(f)

    split_paths = {
        "train": os.path.join(in_path, "NCT-CRC-HE-100K"),
        "val": os.path.join(in_path, "CRC-VAL-HE-7K"),
    }
    if zipped:
        new_in_path = os.path.join(out_path, "..", "NCT-CRC_temp")
        for split, root_path in split_paths.items():
            with ZipFile(f"{root_path}.zip", "r") as zf:
                zf.extractall(new_in_path)
                split_paths[split] = split_paths[split].replace(in_path, new_in_path)
        in_path = new_in_path

    with UnifiedDatasetWriter(out_path, info_path, add_annot_cols=["tissue_class_label"]) as writer:
        for split, root_path in split_paths.items():
            class_to_idx = {re.search(r"\((\w+)\)", v).group(1): k for k, v in info_dict["tasks"][0]["labels"].items()}
            batches = folder_paths(root=root_path, batch_size=batch_size, class_dict=class_to_idx)
            for paths, labs in tqdm(batches, desc=f"Processing NCT-CRC ({split} split)"):
                writer.write(
                    old_paths=[os.path.relpath(p, in_path) for p in paths],
                    original_splits=[split] * len(paths),
                    task_labels=[[lab] for lab in labs],
                    add_annots=[[info_dict["tasks"][0]["labels"][lab]] for lab in labs],
                    images_in_base_path=in_path,
                )

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()
