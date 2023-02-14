"""Saves the NCT-CRC dataset in the unified format.

Expects zip files as downloaded from https://zenodo.org/record/1214456
at ORIGINAL_DATA_PATH/NCT-CRC/NCT-CRC-HE-100K.zip and CRC-VAL-HE-7K.zip if zipped=True,
or extracted folders in ORIGINAL_DATA_PATH/NCT-CRC/NCT-CRC-HE-100K and ORIGINAL_DATA_PATH/NCT-CRC/CRC-VAL-HE-7K if zipped=False.
"""

import os
from shutil import rmtree
from tqdm import tqdm
from torch.utils.data import DataLoader
from zipfile import ZipFile
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter, ImageFolderPaths


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "NCT-CRC"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "NCT-CRC"),
    info_path=os.path.join(INFO_PATH, "NCT-CRC.yaml"),
    batch_size=2048,
    zipped=True,
):
    with UnifiedDatasetWriter(out_path, info_path) as writer:
        # original data separated in train and validation datasets
        for split, root_path in (
            ("train", os.path.join(in_path, "NCT-CRC-HE-100K")),
            ("validation", os.path.join(in_path, "CRC-VAL-HE-7K")),
        ):
            # extract folder
            if zipped:
                new_root_path = os.path.join(out_path, "..", "NCT-CRC_temp")
                with ZipFile(f"{root_path}.zip", 'r') as zf:
                    zf.extractall(new_root_path)
                root_path = new_root_path

            # dummy loader to avoid actually loading the images, since just copied
            dataset = ImageFolderPaths(root=root_path, loader=lambda p: os.path.exists(p))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for _, labs, paths in tqdm(dataloader, desc=f"Processing NCT-CRC ({split} split)"):
                writer.write(
                    old_paths=list(paths),
                    original_splits=[split] * len(paths),
                    task_labels=[[int(lab)] for lab in labs],
                )

            # remove extracted folder to free up space
            if zipped:
                rmtree(root_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()
