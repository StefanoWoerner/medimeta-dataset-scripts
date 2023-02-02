"""Saves the NCT-CRC dataset in the unified format.
"""

import os
from shutil import rmtree
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from zipfile import ZipFile
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter


class ImageFolderPaths(ImageFolder):
    """Modified torchvision.datasets.ImageFolder that returns paths too.
    """
    def __getitem__(self, index):
        img, lab = super(ImageFolderPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, lab, path)


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "NCT-CRC"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "NCT-CRC"),
    info_path=os.path.join(INFO_PATH, "NCT-CRC.yaml"),
    batch_size=64,
):
    with UnifiedDatasetWriter(out_path, info_path) as writer:
        # original data separated in train and validation datasets
        for split, root_path in (
            ("train", os.path.join(in_path, "NCT-CRC-HE-100K")),
            ("validation", os.path.join(in_path, "CRC-VAL-HE-7K")),
        ):
            # extract folder
            with ZipFile(f"{root_path}.zip", 'r') as zf:
                zf.extractall(os.path.join(root_path, ".."))

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
            rmtree(root_path)


if __name__ == "__main__":
    get_unified_data()
